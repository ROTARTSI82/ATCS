/**
 * network.rs
 * By Grant Yang
 * Created on 2023.09.05
 */
use crate::config::{ident, ident_deriv, FuncT, NumT};

/// Training data case with the input and the expected output
#[derive(Debug)]
pub struct TrainCase
{
   pub inp: Box<[NumT]>,
   pub outp: Box<[NumT]>,
}

/// Neural network, containing both layer weights and configuration data
#[derive(Debug)]
pub struct NeuralNetwork
{
   pub layers: Box<[NetworkLayer]>, /// See `NetworkLayer`

   pub activations: Box<[Box<[NumT]>]>, ///
   pub derivs: [Box<[NumT]>; 2], ///

   pub threshold_func: FuncT,
   pub threshold_func_deriv: FuncT,

   pub learn_rate: NumT,
   pub max_width: i32,

   pub max_iters: i32,
   pub err_threshold: NumT,
   pub do_training: bool,

   pub printout_period: i32,
} // pub struct NeuralNetwork

#[derive(Debug)]
pub struct NetworkLayer
{
   pub weights: Box<[NumT]>,
   pub delta_weights: Box<[NumT]>,

   pub num_inputs: i32,
   pub num_outputs: i32,
}

const INPUT_DERIV: usize = 0;
const OUTPUT_DERIV: usize = 1;

impl NetworkLayer
{
   /// Allocates memory for a neural network layer according to the dimensions
   /// of the inputs and outputs. If `train` is true, this function allocates `delta_weights` too.
   pub fn new(num_inputs: i32, num_outputs: i32, train: bool) -> NetworkLayer
   {
      let delta_weights = if train
      {
         vec![0 as NumT; (num_inputs * num_outputs) as usize].into_boxed_slice()
      }
      else
      {
         Box::new([])
      };

      NetworkLayer {
         num_inputs,
         num_outputs,
         weights: vec![0 as NumT; (num_inputs * num_outputs) as usize].into_boxed_slice(),
         delta_weights,
      }
   }

   /**
    * The following functions, `get_weight_mut`, `get_delta_weight_mut`, and `get_weight`
    * are used for indexing into the arrays for constant and mutable values.
    * These make code easier since I used flattened weight arrays. Instead of using a 2d
    * array to store the weight matrices, I used 1d arrays in row-major order.
    */

   pub fn get_weight_mut(&mut self, inp_index: usize, out_index: usize) -> &mut NumT
   {
      assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      &mut self.weights[inp_index * self.num_outputs as usize + out_index]
   }

   pub fn get_delta_weight_mut(&mut self, inp_index: usize, out_index: usize) -> &mut NumT
   {
      assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      &mut self.delta_weights[inp_index * self.num_outputs as usize + out_index]
   }

   pub fn get_weight(&self, inp_index: usize, out_index: usize) -> &NumT
   {
      assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      &self.weights[inp_index * self.num_outputs as usize + out_index]
   }

   /**
    * Feeds the `inp` array forward through this network layer,
    * using the `act` activation function and outputting into the `out` array.
    */
   pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)
   {
      assert!(inp.len() == self.num_inputs as usize && out.len() == self.num_outputs as usize);

      for out_it in 0..self.num_outputs as usize
      {
         let product_part = (0..self.num_inputs as usize)
            .map(|in_it| self.get_weight(in_it, out_it) * inp[in_it])
            .sum::<NumT>();

         out[out_it] = act(product_part);
      }
   } // pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)

   /**
    * Performs backpropagation on this network layer.
    *
    * `inp` is an array of the original inputs fed into this layer, and `outp`
    * is this layer's output values during the feed-forward process.
    *
    * `learn_rate` is the lambda value, the step size for gradient descent.
    *
    * `act_prime` gives the derivative of the threshold function when given
    * the output value of the threshold function.
    * Mathematically, it is f'(f^-1(x)) for the activation function f(x).
    *
    * `deriv_wrt_outp` is the derivative of the cost function with respect to
    * this layer's output.
    *
    * `dest_deriv_wrt_inp` is the array to output into, where this function will write
    * the derivative of the cost function with respect to this layer's input. This value
    * will then be fed into the feed_backwards() function of the previous layer, as
    * this layer's input is the previous layer's output, so `dest_deriv_wrt_inp`
    * becomes the new `deriv_wrt_outp` of the previous layer.
    */
   pub fn feed_backwards(
      &mut self,
      inp: &[NumT],
      outp: &[NumT],
      learn_rate: NumT,
      act_prime: FuncT,
      deriv_wrt_outp: &[NumT],
      dest_deriv_wrt_inp: &mut [NumT],
   )
   {
      assert_eq!(outp.len(), self.num_outputs as usize);
      assert_eq!(inp.len(), self.num_inputs as usize);

      dest_deriv_wrt_inp.fill(0 as NumT);
      for (out_it, act) in outp.iter().enumerate()
      {
         let deriv_wrt_inner = deriv_wrt_outp[out_it] * act_prime(*act);

         for (in_it, wrt_outer) in dest_deriv_wrt_inp.iter_mut().enumerate()
         {
            *wrt_outer += deriv_wrt_inner * self.get_weight(in_it, out_it);

            *self.get_delta_weight_mut(in_it, out_it) += deriv_wrt_inner * inp[in_it] * learn_rate;
         }
      } // for (out_it, act) in outp.iter().enumerate()
   } // pub fn feed_backwards(...)


   pub fn apply_delta_weights(&mut self)
   {
      for (weight, delta) in self.weights.iter_mut().zip(self.delta_weights.iter())
      {
         *weight += delta;
      }

      self.delta_weights.fill(0 as NumT);
   }
} // impl NetworkLayer

impl NeuralNetwork
{
   pub fn new() -> NeuralNetwork
   {
      NeuralNetwork
      {
         layers: Box::new([]),
         activations: Box::new([]),
         threshold_func: ident,
         threshold_func_deriv: ident_deriv,
         learn_rate: 1 as NumT,
         max_width: 0,
         derivs: [Box::new([]), Box::new([])],

         max_iters: 0,
         err_threshold: 0 as NumT,
         do_training: false,
         printout_period: 0,
      } // NeuralNetwork
   } // pub fn new() -> NeuralNetwork

   /// Gets a mutable reference to the input array of the network
   pub fn get_inputs(&mut self) -> &mut [NumT]
   {
      &mut self.activations[0]
   }

   /// Gets the output array of the network. `feed_forward` will write its output here.
   pub fn get_outputs(&self) -> &[NumT]
   {
      &self.activations[self.activations.len() - 1]
   }

   /**
    * Runs the full neural network forwards.
    * The inputs into the network should be written into the array returned by
    * get_inputs(), and the outputs can be read from the array returned by get_outputs().
    */
   pub fn feed_forward(&mut self)
   {
      for (index, layer) in self.layers.iter().enumerate()
      {
         let (input_slice, output_slice) = self.activations.split_at_mut(index + 1);
         layer.feed_forward(
            &input_slice[index],
            &mut output_slice[0],
            self.threshold_func,
         );
      }
   } // pub fn feed_forward(&mut self)

   /**
    * Runs backpropagation through the full neural network.
    * The `expected_out` is the expected output array of the network and is
    * used to compute the cost function and gradient.
    * This function returns the value of the cost function on this particular
    * training case.
    */
   pub fn feed_backwards(&mut self, expected_out: &[NumT]) -> NumT
   {
      assert_eq!(expected_out.len(), self.get_outputs().len());

      let diff = expected_out[0] - self.get_outputs()[0];
      let mut cost = 0.5 * diff * diff;
      self.derivs[INPUT_DERIV][0] = diff;

      for (index, layer) in self.layers.iter_mut().enumerate().rev()
      {
         // rust cannot let us borrow 2 values from a single array at the same time,
         // so we have to use this hack to appease the borrow checker
         let (inp_deriv, outp_deriv) = self.derivs.split_at_mut(1);
         let (inp_act, outp_act) = self.activations.split_at(index + 1);

         layer.feed_backwards(
            &inp_act[index],
            &outp_act[0],
            self.learn_rate,
            self.threshold_func_deriv,
            &inp_deriv[0][..layer.num_outputs as usize],
            &mut outp_deriv[0][..layer.num_inputs as usize],
         );

         self.derivs.swap(INPUT_DERIV, OUTPUT_DERIV);
      }

      cost
   } // pub fn feed_backwards(&mut self, expected_out: &[NumT]) -> NumT

   /// Changes the weights throughout the network according to the stored deltas,
   /// and then clears the deltas to zero for the next backpropagation step.
   pub fn apply_delta_weights(&mut self)
   {
      for layer in self.layers.iter_mut()
      {
         layer.apply_delta_weights();
      }
   }
} // impl NeuralNetwork
