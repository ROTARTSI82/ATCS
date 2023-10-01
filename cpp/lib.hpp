#include <vector>
#include <memory>
#include <iostream>
#include <random>

namespace grad {
    using NumT = double;

    struct Tensor {
        NumT *data;
        NumT *derivs;
        int width, height;
        bool owned = true;

        explicit Tensor(bool diffable, int w, int h = 1) :
                data(new NumT[w * h * (diffable ? 2 : 1)]),
                derivs(diffable ? data + w * h : nullptr),
                width(w), height(h) {
            assert(w > 0);
            assert(h > 0);
            zero_out();
        }

        ~Tensor() {
            if (owned)
                delete[] data;
        }

        [[nodiscard]] inline int tot_size() const {
            return width * height;
        }

        [[nodiscard]] NumT &get(int w, int h = 0) {
            assert(w < width && w >= 0);
            assert(h < height && h >= 0);
            return data[w * (1) + h * (width)];
        };

        [[nodiscard]] NumT &get_deriv(int w, int h = 0) {
            assert(w < width && w >= 0);
            assert(h < height && h >= 0);
            return derivs[w * (1) + h * (width)];
        }

        inline void zero_out() const {
            memset(data, 0, sizeof(NumT) * tot_size());
            if (derivs)
                memset(derivs, 0, sizeof(NumT) * tot_size());
        }

        inline void zero_derivs() const {
            if (derivs)
                memset(derivs, 0, sizeof(NumT) * tot_size());
        }

        void randomize() {
            static std::random_device dev;
            static std::mt19937 rng(dev());
            static std::uniform_real_distribution<NumT> dist(-1.5, 1.5);

            for (int w = 0; w < width; w++)
                for (int h = 0; h < height; h++)
                    get(w, h) = dist(rng);
        }

        void dbg_dump() {
            std::cout << '[';
            for (int r = 0; r < height; r++) {
                std::cout << '[';
                for (int c = 0; c < width; c++)
                    std::cout << get(c, r) << ", ";
                std::cout << "],\n";
            }
            std::cout << "]\n";
        }
    };

    struct ADAMParameter {
        Tensor param;
        Tensor moment;
        Tensor vel;

        explicit ADAMParameter(int w, int h = 1) : param(true, w, h), moment(false, w, h), vel(false, w, h) {}

        constexpr static NumT BETA1 = 0.9;
        constexpr static NumT BETA2 = 0.999;
        constexpr static NumT EPS = 1e-8;
        constexpr static NumT LEARN_RATE = 0.001;
        constexpr static NumT WEIGHT_DECAY = 1e-5;

        inline void apply_grad(int step) {
            NumT alpha = LEARN_RATE * std::sqrt(1.0 - std::pow(BETA2, step)) / (1.0 - std::pow(BETA1, step));

            for (int r = 0; r < param.height; r++)
                for (int c = 0; c < param.width; c++) {
                    NumT grad = param.get_deriv(c, r) + WEIGHT_DECAY * param.get(c, r);
                    NumT &m = moment.get(c, r);
                    m *= BETA1;
                    m += (1.0 - BETA1) * grad;

                    NumT &v = vel.get(c, r);
                    v *= BETA2;
                    v += (1.0 - BETA2) * grad * grad;

                    param.get(c, r) -= alpha * m / (std::sqrt(v) + EPS);
                    param.get_deriv(c, r) = 0;
                }
        }

        [[nodiscard]] inline NumT &get(int w, int h = 0) {
            return param.get(w, h);
        }
    };

    struct Operation {
        virtual ~Operation() = default;

        virtual void forwards() = 0;

        virtual void backwards() = 0;
    };

    struct MatMul final : public Operation {
        Tensor *inp, *mat, *out;
        bool transpose;

        MatMul(Tensor *inp, Tensor *mat, Tensor *out, bool transpose = false) : inp(inp), mat(mat), out(out),
                                                                                transpose(transpose) {
            assert(mat->width == (transpose ? inp->width : inp->height));
            assert(mat->height == out->height);
            assert(out->width == (transpose ? inp->height : inp->width));
        }

        ~MatMul() final = default;

        void forwards() final {
            const int dir = transpose ? inp->height : inp->width;
            for (int inc = 0; inc < dir; inc++)
                for (int r = 0; r < mat->height; r++) {
                    out->get(inc, r) = 0;
                    for (int c = 0; c < mat->width; c++) {
                        const NumT iv = (transpose ? inp->get(c, inc) : inp->get(inc, c));
                        out->get(inc, r) += mat->get(c, r) * iv;
                    }
                }
        }

        void backwards() final {
            const int dir = transpose ? inp->height : inp->width;
            for (int inc = 0; inc < dir; inc++)
                for (int r = 0; r < mat->height; r++) {
                    NumT out_deriv = out->get_deriv(inc, r);
                    for (int c = 0; c < mat->width; c++) {
                        NumT &iv = transpose ? inp->get(c, inc) : inp->get(inc, c);
                        NumT &dv = transpose ? inp->get_deriv(c, inc) : inp->get_deriv(inc, c);
                        dv += mat->get(c, r) * out_deriv;
                        mat->get_deriv(c, r) += iv * out_deriv;
                    }
                }
        }
    };

    struct AddInPlace final : public Operation {
        Tensor *lhs, *rhs;

        AddInPlace(Tensor *lhs, Tensor *rhs) : lhs(lhs), rhs(rhs) {};

        ~AddInPlace() final = default;

        void forwards() final {
            for (int r = 0; r < lhs->height; r++)
                for (int c = 0; c < lhs->width; c++)
                    lhs->get(c, r) += rhs->get(c, r);
        }

        void backwards() final {
            for (int r = 0; r < lhs->height; r++)
                for (int c = 0; c < lhs->width; c++)
                    rhs->get_deriv(c, r) += lhs->get_deriv(c, r);
        }
    };

    struct ElementwiseFunctionInPlace final : public Operation {
        Tensor *tensor;

        explicit ElementwiseFunctionInPlace(Tensor *tensor) : tensor(tensor) {};

        ~ElementwiseFunctionInPlace() final = default;

        void forwards() final {
            for (int r = 0; r < tensor->height; r++)
                for (int c = 0; c < tensor->width; c++) {
                    NumT &x = tensor->get(c, r);
                    x = 1 / (1 + std::exp(-x));
                }
        }

        void backwards() final {
            for (int r = 0; r < tensor->height; r++)
                for (int c = 0; c < tensor->width; c++) {
                    NumT &x = tensor->get(c, r);
                    tensor->get_deriv(c, r) *= x * (1 - x);
                }
        }
    };

    struct Softmax final : public Operation {
        Tensor *in, *out;

        // both input and output are needed :(
        Softmax(Tensor *in, Tensor *out) : in(in), out(out) {};

        ~Softmax() final = default;

        void forwards() final {
            // softmax should be performed by row
            for (int r = 0; r < in->height; r++) {
                NumT accum = 0;
                for (int c = 0; c < in->width; c++)
                    accum += (out->get(c, r) = std::exp(in->get(c, r)));

                for (int c = 0; c < out->width; c++)
                    out->get(c, r) /= accum;
            }
        }

        void backwards() final {
            // i'm pretty sure this works? i have no fucking clue, i'm just trusting that
            // mathematica gave me the correct derivatives.
            for (int r = 0; r < in->height; r++) {
                for (int c = 0; c < in->width; c++) {
                    in->get_deriv(c, r) += out->get(c, r) * out->get_deriv(c, r);
                    for (int c2 = 0; c2 < in->width; c2++) {
                        in->get_deriv(c, r) -= out->get_deriv(c2, r) * out->get(c2, r) * out->get(c, r);
                    }
                }
            }
        }
    };

    struct LayerNormInPlace final : public Operation {
        Tensor *tensor;
        ADAMParameter gain, bias;
        NumT var = 1, mean = 0;

        constexpr static NumT EPS = 1e-05;
        constexpr static NumT BETA_M = 0.51;
        constexpr static NumT BETA_V = 0.01;

        explicit LayerNormInPlace(Tensor *tensor) : tensor(tensor), gain(tensor->width, tensor->height),
                                                    bias(tensor->width, tensor->height) {
            for (int i = 0; i < bias.param.width; i++)
                for (int j = 0; j < bias.param.height; j++) {
                    bias.param.get(i, j) = 0;
                    gain.param.get(i, j) = 1;
                }
        }

        ~LayerNormInPlace() final = default;

        void forwards() final {
            NumT cur_mean = 0;
            NumT cur_var = 0;
            for (int r = 0; r < tensor->height; r++) {
                for (int c = 0; c < tensor->width; c++) {
                    NumT &x = tensor->get(c, r);
                    cur_mean += x;
                    cur_var += (x - mean) * (x - mean);
                }
            }

            cur_mean /= tensor->tot_size();
            cur_var /= tensor->tot_size();
            mean = BETA_M * mean + (1 - BETA_M) * cur_mean;
            var = BETA_V * var + (1 - BETA_V) * cur_var;

            for (int r = 0; r < tensor->height; r++) {
                for (int c = 0; c < tensor->width; c++) {
                    NumT &x = tensor->get(c, r);
                    x = gain.get(c, r) * (x - mean) / (var + EPS) + bias.get(c, r);
                }
            }
        }

        void backwards() final {
            for (int r = 0; r < tensor->height; r++)
                for (int c = 0; c < tensor->width; c++) {
                    NumT &deriv = tensor->get_deriv(c, r);
                    gain.param.get_deriv(c, r) += deriv * (tensor->get(c, r) - mean) / (var + EPS);
                    bias.param.get_deriv(c, r) += deriv;
                    deriv *= gain.get(c, r) / (var + EPS);
                }
        }

        inline void apply_grad(int step) {
            gain.apply_grad(step);
            bias.apply_grad(step);
        }
    };
}