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

        explicit Tensor(int w, int h = 1, int cw = 1, int ch = 1) : data(new NumT[w * h * cw * ch]), shape{w, h, cw, ch} {
            assert(w > 0);
            assert(h > 0);
            assert(cw > 0);
            assert(ch > 0);
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
            static std::uniform_real_distribution<NumT> dist(-1.5,1.5);

            for (int w = 0; w < shape.width; w++)
                for (int h = 0; h < shape.height; h++)
                    for (int cw = 0; cw < shape.chan_width; cw++)
                        for (int ch = 0; ch < shape.chan_height; ch++)
                            get(w, h, cw, ch) = dist(rng);
        }
    };

    struct ADAMParameter {
        Tensor param;
        Tensor moment;
        Tensor vel;

        explicit ADAMParameter(int w, int h = 1, int cw = 1, int ch = 1) : param{w, h, cw, ch}, moment{w, h, cw, ch}, vel{w, h, cw, ch} {}

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
        Tensor output;

        // derivative of the cost function with respect to this layer's output, written to by
        // our dependent layers during backpropagation
        Tensor d_wrt_out;

        // layers that we depend on for input
        std::vector<std::shared_ptr<Layer>> dependencies;

        explicit Layer(int w, int h = 1, int cw = 1, int ch = 1) : output{w, h, cw, ch}, d_wrt_out{w, h, cw, ch} {}
        virtual ~Layer() = default;

        virtual void recursive_forwards() = 0;
        virtual void recursive_backwards(int step) = 0;
        virtual void recursive_randomize() = 0;

        [[nodiscard]] inline NumT &get(int w, int h = 0, int cw = 0, int ch = 0) const {
            return output.get(w, h, cw, ch);
        }
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
        explicit InputLayer(int w, int h = 1, int cw = 1, int ch = 1) : Layer(w, h, cw, ch) {};
        ~InputLayer() final = default;

        void recursive_forwards() final {
//            std::cout << "input layer forward\n\n";
        }
        void recursive_backwards(int) final {
//            std::cout << "input recursive_backwards\n";
        }

        void recursive_randomize() final {
            output.randomize();
        }
    };

    struct DenseLayer final : public Layer {
        ADAMParameter weights;
        ADAMParameter biases;

        DenseLayer(const std::shared_ptr<Layer> &prev, int w) : Layer(w), weights{prev->output.tot_size(), w}, biases(w) {
            dependencies.push_back(prev);
        }
        ~DenseLayer() final = default;

        void recursive_randomize() final {
            dependencies[0]->recursive_randomize();
            weights.param.randomize();
            biases.param.randomize();
        }

        void recursive_forwards() final {
            dependencies[0]->recursive_forwards();

            Tensor &input = dependencies[0]->output;
            for (int out = 0; out < output.shape.width; out++) {
                output.get(out) = biases.get(out);
                for (int inp = 0; inp < input.shape.width; inp++) {
                    output.get(out) += input.get(inp) * weights.get(inp, out);
                }

                output.get(out) = activation_func(output.get(out));
            }
        }

        void recursive_backwards(int step) final {
            NumT alpha = ADAMParameter::get_alpha(step);

            Tensor &input = dependencies[0]->output;
            Tensor &d_wrt_inp = dependencies[0]->d_wrt_out;

            // save the shape and temporarily flatten it
            TensorShape orig_shape = input.shape;
            assert(weights.param.shape.width == input.tot_size() && input.tot_size() == d_wrt_inp.tot_size());
            d_wrt_inp.shape = input.shape = {weights.param.shape.width, 1, 1, 1};

            for (int out = 0; out < output.shape.width; out++) {
                NumT d_wrt_inner = d_wrt_out.get(out) * activation_prime_of_inv(output.get(out));

                for (int in = 0; in < input.shape.width; in++) {
                    d_wrt_inp.get(in) += d_wrt_inner * weights.get(in, out);
                    weights.update_with(alpha, input.get(in) * d_wrt_inner, in, out);
                }

                biases.update_with(alpha, d_wrt_inner, out);
            }

            d_wrt_inp.shape = (input.shape = orig_shape);

            d_wrt_out.zero_out();
            dependencies[0]->recursive_backwards(step);
        }
    };

    struct ConvolutionPoolLayer final : public Layer {
        void recursive_forwards() final {

        };

        void recursive_backwards(int step) final {

        };

        void recursive_randomize() final {

        };
    };
}