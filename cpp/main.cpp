//
// Created by Grant Yang on 9/25/23.
//

#include "lib.hpp"
#include <iostream>

using namespace grad;

int main() {
    auto inputs = std::make_shared<InputLayer>(2);
    auto h1 = std::make_shared<DenseLayer>(inputs, 4);
    auto outp = std::make_shared<DenseLayer>(h1, 1);
    outp->recursive_randomize();

    int step = 1, epoch;
    for (epoch = 0; epoch < 40000; epoch++) {
        NumT cost = 0;
        for (int c = 0; c < 4; c++) {
            int a = c & 1;
            int b = (c & 2) >> 1;

            inputs->get(0) = (NumT) a;
            inputs->get(1) = (NumT) b;

            outp->recursive_forwards();

            cost += 0.5 * pow(outp->get(0) - (a ^ b), 2);

            if (epoch % 100 == 0)
                std::cout << "net " << a << ", " << b << " = " << outp->get(0) << '\n';

            outp->get_deriv(0) = outp->get(0) - (a ^ b);
            outp->recursive_backwards(step++);
        }

        if (cost < 0.002)
            break;

        if (epoch % 100 == 0) {
            std::cout << "Cost = " << cost << "\n\n\n";
        }
    }

    std::cout << "hi, stopped after " << epoch << " epochs\n";
    NumT cost = 0;
    for (int c = 0; c < 4; c++) {
        int a = c & 1;
        int b = (c & 2) >> 1;

        inputs->get(0) = (NumT) a;
        inputs->get(1) = (NumT) b;

        outp->recursive_forwards();

        cost += 0.5 * pow(outp->get(0) - (a ^ b), 2);
        std::cout << "net " << a << ", " << b << " = " << outp->get(0) << '\n';
    }
    std::cout << "cost = " << cost << '\n';
}
