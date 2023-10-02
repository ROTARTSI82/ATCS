//
// Created by Grant Yang on 10/1/23.
//

#include "lib.hpp"
#include <ranges>
#include <vector>

namespace grad {
    struct AttnHead {
        Tensor q_mat, k_mat, v_mat;
        Tensor attn_mat, attn_softmaxed;
        ADAMParameter q_proj, k_proj, v_proj;

        inline void grad_and_zero(uint64_t step) {
            q_proj.apply_grad(step);
            k_proj.apply_grad(step);
            v_proj.apply_grad(step);

            attn_mat.zero_derivs();
            attn_softmaxed.zero_derivs();
            q_mat.zero_derivs();
            k_mat.zero_derivs();
            v_mat.zero_derivs();
        }
    };

    struct SelfAttn {
        Tensor inputs;
        AttnHead *heads;
        ADAMParameter out_proj;
        Tensor attn_final;
        Tensor output;

        Tensor *references = nullptr;
        std::vector<std::unique_ptr<Operation>> ops;
        int num_heads;

        // v_size * n_heads should be about num_chan, and qk_size should be about v_size.
        SelfAttn(int num_chan, int num_tok, int qk_size, int v_size, int n_heads) :
                inputs(true, num_chan, num_tok),
                out_proj(v_size * n_heads, num_chan),
                attn_final(true, v_size * n_heads, num_tok),
                output(true, num_chan, num_tok), num_heads(n_heads) {

            references = new Tensor[(inputs.height * 4 + 1) * n_heads + 2 * inputs.height];
            heads = new AttnHead[n_heads];
            auto r = references;

            for (int head = 0; head < n_heads; head++) {
                heads[head].q_mat = Tensor(true, qk_size, num_tok);
                heads[head].k_mat = Tensor(true, qk_size, num_tok);
                heads[head].v_mat = Tensor(true, v_size, num_tok);
                heads[head].q_proj = ADAMParameter(num_chan, qk_size);
                heads[head].k_proj = ADAMParameter(num_chan, qk_size);
                heads[head].v_proj = ADAMParameter(num_chan, v_size);
                heads[head].attn_mat = Tensor(true, num_tok, num_tok);
                heads[head].attn_softmaxed = Tensor(true, num_tok, num_tok);

                for (int tok = 0; tok < inputs.height; tok++) {
                    auto &ref = *r++ = Tensor(&inputs, 0, tok, inputs.width);
                    auto &q_ref = *r++ = Tensor(&heads[head].q_mat, 0, tok, heads[head].q_mat.width);
                    auto &k_ref = *r++ = Tensor(&heads[head].k_mat, 0, tok, heads[head].k_mat.width);
                    auto &v_ref = *r++ = Tensor(&heads[head].v_mat, 0, tok, heads[head].v_mat.width);

                    ops.emplace_back(std::make_unique<MatMul>(&ref, &heads[head].q_proj.param, &q_ref, true, true));
                    ops.emplace_back(std::make_unique<MatMul>(&ref, &heads[head].k_proj.param, &k_ref, true, true));
                    ops.emplace_back(std::make_unique<MatMul>(&ref, &heads[head].v_proj.param, &v_ref, true, true));
                }

                ops.emplace_back(std::make_unique<MatMul>(&heads[head].k_mat, &heads[head].q_mat, &heads[head].attn_mat, true, false, 1/std::sqrt(qk_size)));
                ops.emplace_back(std::make_unique<Softmax>(&heads[head].attn_mat, &heads[head].attn_softmaxed));

                auto &final_ref = *r++ = Tensor(&attn_final, head * v_size, 0, v_size);
                final_ref.height = num_tok;
                ops.emplace_back(std::make_unique<MatMul>(&heads[head].v_mat, &heads[head].attn_softmaxed, &final_ref));
            }

            for (int tok = 0; tok < inputs.height; tok++) {
                auto &ref = *r++ = Tensor(&attn_final, 0, tok, attn_final.width);
                auto &out_ref = *r++ = Tensor(&output, 0, tok, output.width);
                ops.emplace_back(std::make_unique<MatMul>(&ref, &out_proj.param, &out_ref, true, true));
            }
        };

        ~SelfAttn() {
            delete[] references;
            delete[] heads;
        }

        void forwards() {
            for (auto &op: ops)
                op->forwards();
        }

        void backwards() {
            for (auto &op: std::ranges::reverse_view(ops))
                op->backwards();
        }

        void grad_and_reset(uint64_t step) {
            for (int h = 0; h < num_heads; h++)
                heads[h].grad_and_zero(step);

            out_proj.apply_grad(step);

            attn_final.zero_derivs();
            output.zero_derivs();
        }
    };

    struct Transformer {

        void forwards() {
        }

        void backwards() {
        }
    };
}
