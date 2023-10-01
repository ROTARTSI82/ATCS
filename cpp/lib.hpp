#include <vector>
#include <memory>
#include <random>

namespace grad {
    using NumT = double;

    struct TensorShape {
        int width = 1, height = 1, chan_width = 1, chan_height = 1;
    };

    struct Tensor {
        std::unique_ptr<NumT[]> data;
        TensorShape shape;

        static Tensor TRASH;

        explicit Tensor(int w, int h = 1, int cw = 1, int ch = 1) : data(new NumT[w * h * cw * ch]),
                                                                    shape{w, h, cw, ch} {
            assert(w > 0);
            assert(h > 0);
            assert(cw > 0);
            assert(ch > 0);
            zero_out();
        }

        explicit Tensor(const TensorShape &shape) : shape(shape) {
            zero_out();
        }

        [[nodiscard]] inline int tot_size() const {
            return shape.width * shape.height * shape.chan_height * shape.chan_width;
        }

        [[nodiscard]] NumT &get(int w, int h = 0, int cw = 0, int ch = 0) const {
            assert(w < shape.width && w >= 0);
            assert(h < shape.height && h >= 0);
            assert(cw < shape.chan_width && cw >= 0);
            assert(ch < shape.chan_height && ch >= 0);
            return data[w + h * shape.width
                        + cw * shape.width * shape.height
                        + ch * shape.width * shape.height * shape.chan_width];
        };

        inline void zero_out() const {
            memset(data.get(), 0, sizeof(NumT) * tot_size());
        }

        void randomize() const {
            static std::random_device dev;
            static std::mt19937 rng(dev());
            static std::uniform_real_distribution<NumT> dist(-1.5, 1.5);

            for (int w = 0; w < shape.width; w++)
                for (int h = 0; h < shape.height; h++)
                    for (int cw = 0; cw < shape.chan_width; cw++)
                        for (int ch = 0; ch < shape.chan_height; ch++)
                            get(w, h, cw, ch) = dist(rng);
        }
    };

    Tensor Tensor::TRASH = Tensor{1};

    struct ADAMParameter {
        Tensor param;
        Tensor moment;
        Tensor vel;

        explicit ADAMParameter(const TensorShape &shape) : param{shape}, moment{shape}, vel{shape} {}

        explicit ADAMParameter(int w, int h = 1, int cw = 1, int ch = 1) : param{w, h, cw, ch}, moment{w, h, cw, ch},
                                                                           vel{w, h, cw, ch} {}

        constexpr static NumT BETA1 = 0.9;
        constexpr static NumT BETA2 = 0.999;
        constexpr static NumT EPS = 1e-8;
        constexpr static NumT LEARN_RATE = 0.3; // 0.001;

        [[nodiscard]] inline static NumT get_alpha(int step) {
            return LEARN_RATE * std::sqrt(1.0 - std::pow(BETA2, step)) / (1.0 - std::pow(BETA1, step));
        }

        inline void update_with(NumT alpha, NumT grad, int w, int h = 0, int cw = 0, int ch = 0) const {
            NumT &m = moment.get(w, h, cw, ch);
            m *= BETA1;
            m += (1.0 - BETA1) * grad;

            NumT &v = vel.get(w, h, cw, ch);
            v *= BETA2;
            v += (1.0 - BETA2) * grad * grad;

            param.get(w, h, cw, ch) -= alpha * m / (std::sqrt(v) + EPS);
        }

        [[nodiscard]] inline NumT &get(int w, int h = 0, int cw = 0, int ch = 0) const {
            return param.get(w, h, cw, ch);
        }
    };

    struct Layer {
        // layers that we depend on for input
//        std::vector<std::shared_ptr<Layer>> dependencies;

        Layer() = default;

        virtual ~Layer() = default;

        virtual void recursive_forwards() = 0;

        virtual void recursive_backwards(int step) = 0;

        virtual void recursive_randomize() = 0;

        [[nodiscard]] inline NumT &get(int w, int h = 0, int cw = 0, int ch = 0) {
            return get_output().get(w, h, cw, ch);
        }

        [[nodiscard]] inline NumT &get_deriv(int w, int h = 0, int cw = 0, int ch = 0) {
            return get_d_wrt_output().get(w, h, cw, ch);
        }

        virtual Tensor &get_output() = 0;

        virtual Tensor &get_d_wrt_output() = 0;
    };

    inline NumT activation_func(NumT x) {
//        return x > 0 ? x : 0.001 * x;
        return 1 / (1 + std::exp(-x));
    }

    inline NumT activation_prime_of_inv(NumT x) {
//        return x > 0 ? 1 : 0.001;
        return x * (1.0 - x);
    }

    struct InputLayer final : public Layer {
        Tensor data;
        Tensor d_wrt_data;

        Tensor &get_output() final { return data; }

        Tensor &get_d_wrt_output() final { return d_wrt_data; }

        explicit InputLayer(int w, int h = 1, int cw = 1, int ch = 1) : data(w, h, cw, ch), d_wrt_data(w, h, cw, ch) {};

        ~InputLayer() final = default;

        void recursive_forwards() final {
            d_wrt_data.zero_out();
        }

        void recursive_backwards(int) final {}

        void recursive_randomize() final {
            data.randomize();
        }
    };

    struct DenseLayer final : public Layer {
        ADAMParameter weights;
        ADAMParameter biases;

        std::shared_ptr<Layer> prev;
        Tensor output;
        Tensor d_wrt_output;

        DenseLayer(const std::shared_ptr<Layer> &prev, int w) : weights{prev->get_output().tot_size(), w}, biases(w),
                                                                prev(prev), output(w), d_wrt_output(w) {}

        ~DenseLayer() final = default;

        Tensor &get_output() final { return output; }

        Tensor &get_d_wrt_output() final { return d_wrt_output; }

        void recursive_randomize() final {
            prev->recursive_randomize();
            weights.param.randomize();
            biases.param.randomize();
        }

        void recursive_forwards() final {
            prev->recursive_forwards();

            Tensor &input = prev->get_output();
            for (int out = 0; out < output.shape.width; out++) {
                output.get(out) = biases.get(out);
                for (int inp = 0; inp < input.shape.width; inp++) {
                    output.get(out) += input.get(inp) * weights.get(inp, out);
                }

                output.get(out) = activation_func(output.get(out));
            }

            // prepare for backpropagation step
            d_wrt_output.zero_out();
        }

        void recursive_backwards(int step) final {
            NumT alpha = ADAMParameter::get_alpha(step);

            Tensor &input = prev->get_output();
            Tensor &d_wrt_inp = prev->get_d_wrt_output();

            // save the shape and temporarily flatten it
            TensorShape orig_shape = input.shape;
            assert(weights.param.shape.width == input.tot_size() && input.tot_size() == d_wrt_inp.tot_size());
            d_wrt_inp.shape = input.shape = {weights.param.shape.width, 1, 1, 1};

            for (int out = 0; out < output.shape.width; out++) {
                NumT d_wrt_inner = d_wrt_output.get(out) * activation_prime_of_inv(output.get(out));

                for (int in = 0; in < input.shape.width; in++) {
                    d_wrt_inp.get(in) += d_wrt_inner * weights.get(in, out);
                    weights.update_with(alpha, input.get(in) * d_wrt_inner, in, out);
                }

                biases.update_with(alpha, d_wrt_inner, out);
            }

            d_wrt_inp.shape = (input.shape = orig_shape);
            prev->recursive_backwards(step);
        }
    };

    struct Convolution2DLayer final : public Layer {
        Tensor output;
        Tensor d_wrt_output;

        ADAMParameter kernel;
        ADAMParameter biases;

        std::shared_ptr<Layer> prev;
        int stride, pad;

        Convolution2DLayer(const std::shared_ptr<Layer> &prev, int outp_chan, int kern_size, int stride, int pad) :
                output(prev->get_output().shape.width - kern_size + 1, prev->get_output().shape.height - kern_size + 1,
                       outp_chan),
                d_wrt_output(output.shape),
                kernel(kern_size, kern_size, prev->get_output().shape.chan_width, outp_chan),
                biases(kernel.param.shape),
                prev(prev), stride(stride), pad(pad) {}

        Tensor &get_output() final {
            return Tensor::TRASH;
        }

        Tensor &get_d_wrt_output() final {
            return Tensor::TRASH;
        }

        void recursive_forwards() final {
            prev->recursive_forwards();
            Tensor &input = prev->get_output();

            int kern_w = kernel.param.shape.width;
            int kern_h = kernel.param.shape.height;

            for (int w = -pad; w <= input.shape.width - kern_w + pad; w += stride) {
                for (int h = -pad; h <= input.shape.height - kern_h + pad; h += stride) {
                    for (int co = 0; co < kernel.param.shape.chan_height; co++) {
                        output.get(w + pad, h + pad, co) = biases.get(w + pad, h + pad, co);
                        for (int kw = 0; kw < kern_w; kw++) {
                            for (int kh = 0; kh < kern_h; kh++) {
                                for (int c = 0; c < input.shape.chan_width; c++) {
                                    NumT inp_v = 0;
                                    if (w + kw >= 0 && w + kw < input.shape.width && h + kh >= 0 &&
                                        h + kh < input.shape.height)
                                        inp_v = input.get(w + kw, h + kh, c);
                                    output.get(w + pad, h + pad, co) += kernel.get(kw, kh, c, co) * inp_v;
                                }
                            }
                        }
                        output.get(w + pad, h + pad, co) = activation_func(output.get(w + pad, h + pad, co));
                    }
                }
            }

            d_wrt_output.zero_out();
        };

        void recursive_backwards(int step) final {
            NumT alpha = ADAMParameter::get_alpha(step);

            prev->recursive_backwards(step);
        };

        void recursive_randomize() final {
            prev->recursive_randomize();
            kernel.param.randomize();
            biases.param.randomize();
        };
    };

    struct PoolLayer2D final : public Layer {
        Tensor output;
        Tensor d_wrt_output;
        std::shared_ptr<Layer> prev;

        PoolLayer2D(const std::shared_ptr<Layer> &prev, int pool_factor, int pool_stride) :
                output((prev->get_output().shape.width + pool_factor - 1) / pool_factor,
                       (prev->get_output().shape.height + pool_factor - 1) / pool_factor),
                d_wrt_output(output.shape),
                prev(prev) {}

        Tensor &get_output() final {
            return Tensor::TRASH;
        }

        Tensor &get_d_wrt_output() final {
            return Tensor::TRASH;
        }

        void recursive_forwards() final {

        };

        void recursive_backwards(int step) final {

        };

        void recursive_randomize() final {

        };
    };
}