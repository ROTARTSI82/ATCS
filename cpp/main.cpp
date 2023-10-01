//
// Created by Grant Yang on 9/25/23.
//

#include "lib.hpp"
#include <iostream>

using namespace grad;

int main() {
    Tensor input{true, 2};
    Tensor hid1{true, 1, 4};
    Tensor hidtrans{true, 4, 1};
    Tensor hidsm{true, 4, 1};
    Tensor hidtrans2{true, 1, 4};
    Tensor out{true, 1, 1};

    ADAMParameter hid1weights{2, 4};
    ADAMParameter outweights{4, 1};
    ADAMParameter hidbias{1,4};
    ADAMParameter outbias{1,1};
    outbias.param.randomize();
    hid1weights.param.randomize();
    outweights.param.randomize();
    hidbias.param.randomize();


    ADAMParameter t1{1,1};
    t1.param.randomize();

    ADAMParameter t2{4,4};
    t2.param.randomize();


    MatMul hidmul{&input, &hid1weights.param, &hid1, true};
    AddInPlace add{&hid1, &hidbias.param};
    LayerNormInPlace norm{&hid1};
    ElementwiseFunctionInPlace activ{&hid1};
    MatMul trans1{&hid1, &t1.param, &hidtrans, true};
    Softmax softmax{&hidtrans, &hidsm};
    MatMul trans2{&hidsm, &t2.param, &hidtrans2, true};
    MatMul outmul{&hidtrans2, &outweights.param, &out};
    AddInPlace add2{&out, &outbias.param};
    ElementwiseFunctionInPlace outactiv{&out};

    int step = 1, epoch;
    for (epoch = 0; epoch < 40000; epoch++) {
        NumT cost = 0;
        for (int c = 0; c < 4; c++) {
            int a = c & 1;
            int b = (c & 2) >> 1;

            input.get(0) = (NumT) a;
            input.get(1) = (NumT) b;

            hidmul.forwards();
            add.forwards();
            norm.forwards();
            activ.forwards();
            trans1.forwards();
            softmax.forwards();
            trans2.forwards();
            outmul.forwards();
            add2.forwards();
            outactiv.forwards();

            cost += 0.5 * pow(out.get(0,0) - (a ^ b), 2);

            if (epoch % 100 == 0)
                std::cout << "net " << a << ", " << b << " = " << out.get(0,0) << '\n';

            out.get_deriv(0,0) += (out.get(0,0) - (a ^ b));

            outactiv.backwards();
            add2.backwards();
            outmul.backwards();
            trans2.backwards();
            softmax.backwards();
            trans1.backwards();
            activ.backwards();
            norm.backwards();
            add.backwards();
            hidmul.backwards();

            hid1weights.apply_grad(step);
            hidbias.apply_grad(step);
            outweights.apply_grad(step);
            outbias.apply_grad(step);

            t1.apply_grad(step);
            t2.apply_grad(step);
            norm.apply_grad(step);

            input.zero_derivs();
            out.zero_derivs();
            hid1.zero_derivs();
            hidtrans.zero_derivs();
            hidtrans2.zero_derivs();
            hidsm.zero_derivs();

            step++;
        }

        if (epoch % 100 == 0) {
//            norm.gain.param.dbg_dump();
//            norm.bias.param.dbg_dump();
            std::cout << "norm mean = " << norm.mean << ", var = " << norm.var << '\n';
            std::cout << "Cost = " << cost << "\n\n\n";
        }

        if (cost < 0.002) {
            std::cout << "\nbreaking after " << epoch << " epochs\n";
            if (epoch == 100)
                break;
            epoch = 99; // hack to get it to print out the thingy
        }
    }
}
