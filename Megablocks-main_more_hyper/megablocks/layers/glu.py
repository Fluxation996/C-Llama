from megablocks.layers import common
from megablocks.layers import gelu,swish
from megablocks.layers.mlp import SparseMLP, create_dmoe_expert_weights
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, InitFn
from megablocks import grouped_gemm_util as gg
import stk
import torch
import torch.nn.functional as F
from megablocks.layers import weight_parallel as wp


class ScaleGradient(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None
scale_gradient = ScaleGradient.apply

class SparseGLU(SparseMLP):

    def __init__(self, args : Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        with torch.no_grad():
            self.v1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))

        mpu.set_expert_model_parallel_attributes(
            self.v1, self._should_set_parallelism_attribute)

        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Weight parallelism not yet supported with GLU.")
        elif self.args.memory_optimized_mlp:
            raise NotImplementedError("Memory optimized implementation not yet supported with GLU.")

    def forward(self, x, topo):
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))

        # Compute the GLU.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)
        if self.args.swiglu:
            x1 = stk.ops.mul(swish.swish(x1), x2)
        else:
            x1 = stk.ops.mul(gelu.gelu(x1), x2)
        return stk.ops.dsd(x1, w2)

class SparseGLU_Llama(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        if self.args.swiglu:
            ffn_hidden_size = args.ffn_hidden_size*2
            self._num_rows_per_rank_w1 = (
                    (mpu.experts_per_rank(args) * mpu.features_per_rank(args))*2 //
                    mpu.get_weight_parallel_world_size(args)
            )
        else:
            ffn_hidden_size = args.ffn_hidden_size
            self._num_rows_per_rank_w1 = (
                    (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
                    mpu.get_weight_parallel_world_size(args)
            )
        self._num_rows_per_rank_w2 = (
                (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
                mpu.get_weight_parallel_world_size(args)
        )

        self.w1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank_w1,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank_w2,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))

        with torch.no_grad():
            self.w1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, ffn_hidden_size,
                args.hidden_size, args.init_method))
            self.w2.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

        self._should_set_parallelism_attribute = (
            args.moe_expert_model_parallelism or args.moe_weight_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w1, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, self._should_set_parallelism_attribute)

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Weight parallelism not yet supported with GLU.")
        elif self.args.memory_optimized_mlp:
            raise NotImplementedError("Memory optimized implementation not yet supported with GLU.")

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)


    #unfinished testing
    def parallel_forward(self, x, topo):
        group = self.args.weight_parallel_group
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Compute the MLP.
        w1 = torch.chunk(w1, 2, dim=0)
        x1 = wp.sdd_nt(x, w1[0], topo, group)
        x2 = wp.sdd_nt(x, w1[1], topo, group)

        if self.args.swiglu:
            x = stk.ops.mul(swish.swish(x1), x2)
        else:
            x = stk.ops.mul(gelu.gelu(x1), x2)


        return wp.dsd_nn(x, w2, group)


    def forward(self, x, topo):
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))
        if self.args.moe_weight_parallelism:
            return self.parallel_forward(x, topo)

        # Compute the GLU.
        #x1 = stk.ops.sdd(x, w1.t(), topo)
        #x2 = stk.ops.sdd(x, v1.t(), topo)
        #Llama style swiglu

        w1 = torch.chunk(w1, 2, dim=0)
        x1 = stk.ops.sdd(x, w1[0].t(), topo)
        x2 = stk.ops.sdd(x, w1[1].t(), topo)

        if self.args.swiglu:
            x1 = stk.ops.mul(swish.swish(x1), x2)
        else:
            x1 = stk.ops.mul(gelu.gelu(x1), x2)
        return stk.ops.dsd(x1, w2)

class GroupedGLU(SparseGLU):
    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        v1 = v1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        # Compute the MLP.
        x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)
        if self.args.swiglu:
            x1 = F.silu(x1) * x2
        else:
            x1 = F.gelu(x1, approximate="tanh") * x2
        return gg.ops.gmm(x1, w2, batch_sizes)


class GroupedGLU_Llama(SparseGLU_Llama):
    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        # Compute the MLP.
        #x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        #x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)

        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = torch.chunk(x, 2, dim=-1)

        if self.args.swiglu:
            x = F.silu(x[0]) * x[1]
        else:
            x = F.gelu(x[0], approximate="tanh") * x[1]
        return gg.ops.gmm(x, w2, batch_sizes)
