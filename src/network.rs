/**
 * network.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 *
 */
use crate::config::{ident, ident_deriv, FuncT, NumT};

/**
 * These are array indices into `NeuralNetwork::activations`,
 * specifying where to read from and where to write to during the
 * backpropagation step. See `NeuralNetwork::feed_backwards` and `NetworkLayer::feed_backwards`
 */
const INPUT_DERIV: usize = 0;
const OUTPUT_DERIV: usize = 1;

/// Training data case with the input and the expected output
#[derive(Debug)]
pub struct TrainCase
{
   pub inputs: Box<[NumT]>,
   pub expected_outputs: Box<[NumT]>,
}

/// Neural network, containing both layer weights and configuration data
#[derive(Debug)]
pub struct NeuralNetwork
{
   /**
    * Stores the weight matrix (and possibly delta weights) of each
    * layer, along with the layer's dimensions. See `NetworkLayer` for more information.
    */
   pub layers: Box<[NetworkLayer]>,

   /**
    * This array stores the input, hidden, and output activations of the network.
    * Index 0 is an array of the inputs to the network, and the last array
    * is the output. Each of the arrays in the middle contain the hidden
    * layers' activations. These values are needed for backpropagation.
    */
   pub activations: Box<[Box<[NumT]>]>,

   /**
    * This array is used to store the partial derivatives
    * needed for backpropagation. Both arrays are of the same size,
    * that size being equal to "fattest" point of the network
    * (i.e. the maximum number of activations for a layer in the network).
    *
    * This is just used as scratch space during backpropagation,
    * and the values stored here are not of any interest.
    */
   pub derivs: [Box<[NumT]>; 2],

   /// The activation functions apply element-wise to each layer's output.
   pub threshold_func: FuncT,
   pub threshold_func_deriv: FuncT,

   /**
    * The following values are copied directly from
    * the configuration file and have the same names.
    * For more information, see the specification of the configuration format
    * in the documentation for `set_and_echo_config` in `config.rs`.
    */
   pub learn_rate: NumT,
   pub error_cutoff: NumT,
   pub max_iterations: i32,
   pub printout_period: i32,
   pub do_training: bool,
} // pub struct NeuralNetwork

/// Represents an individual hidden (or output) layer in the network.
/// Stores both the weight array and, in the case of training, the delta weights.
#[derive(Debug)]
pub struct NetworkLayer
{
   pub weights: Box<[NumT]>,
   pub delta_weights: Box<[NumT]>,

   pub num_inputs: i32,
   pub num_outputs: i32,
}

impl NetworkLayer
{
   /**
    * Allocates memory for a neural network layer according to the dimensions
    * of the inputs and outputs specified by `num_inputs` and `num_outputs`.
    * If `do_train` is true, this function also allocates a `delta_weights` array.
    *
    * This function fills the fields of the network with dummy values.
    * See `set_and_echo_config` in `config.rs` for the code to initialize the network.
    */
   pub fn new(num_inputs: i32, num_outputs: i32, do_training: bool) -> NetworkLayer
   {
      let delta_weights = if do_training
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
    *
    * All of these functions have the precondition of 0 <= `inp_index` < `self.num_inputs`
    * and 0 <= `out_index` < `self.num_outputs`.
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
    * Feeds the `inp_act_arr` input activation array forward through this network layer,
    * using the `threshold_func` activation function.
    * Outputs the results into the `dest_h_out` array.
    *
    * Precondition: `inp_act_arr` must have length equal to `self.num_inputs`
    * and `dest_h_out` must have length equal to `self.num_outputs`.
    */
   pub fn feed_forward(&self, inp_act_arr: &[NumT], dest_h_out: &mut [NumT], threshold_func: FuncT)
   {
      assert!(
         inp_act_arr.len() == self.num_inputs as usize
            && dest_h_out.len() == self.num_outputs as usize
      );

      for out_it in 0..self.num_outputs as usize
      {
         let theta = (0..self.num_inputs as usize)
            .map(|in_it| self.get_weight(in_it, out_it) * inp_act_arr[in_it])
            .sum::<NumT>();

         dest_h_out[out_it] = threshold_func(theta);
      }
   } // pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)

   /**
    * Performs backpropagation on this network layer. No weights are changed,
    * only written to the `delta_weights` array.
    * To apply the changes, call `apply_delta_weights`.
    *
    * `inp_acts_arr` is an array of the original inputs fed into this layer, and `out_acts_arr`
    * is this layer's output values during the feed-forward process.
    * Their lengths MUST be equal to `self.num_inputs` and `self.num_outputs` respectively.
    *
    * `learn_rate` is the lambda value, the step size for gradient descent.
    *
    * `threshold_func_prime` gives the derivative of the threshold function when given
    * the output value of the threshold function.
    * Mathematically, it is f'(f^-1(x)) for the activation function f(x).
    *
    * `deriv_wrt_outp` is the derivative of the cost function with respect to
    * each of this layer's outputs. Its length MUST be equal to `self.num_outputs`
    *
    * `dest_deriv_wrt_inp` is an array to output into, where this function will write
    * the derivative of the cost function with respect to each of this layer's inputs.
    * Its length MUST be equal to `self.num_inputs`.
    * This value will then be fed into the `feed_backwards` function of the previous layer.
    * This layer's input is the previous layer's output, so this layer's `dest_deriv_wrt_inp`
    * becomes the new `deriv_wrt_outp` of the previous layer.
    */
   pub fn feed_backwards(
      &mut self,
      inp_acts_arr: &[NumT],
      out_acts_arr: &[NumT],
      learn_rate: NumT,
      threshold_func_prime: FuncT,
      deriv_wrt_out: &[NumT],
      dest_deriv_wrt_inp: &mut [NumT],
   )
   {
      assert_eq!(out_acts_arr.len(), self.num_outputs as usize);
      assert_eq!(inp_acts_arr.len(), self.num_inputs as usize);
      assert_eq!(deriv_wrt_out.len(), self.num_outputs as usize);
      assert_eq!(dest_deriv_wrt_inp.len(), self.num_inputs as usize);

      dest_deriv_wrt_inp.fill(0 as NumT);
      for (out_it, out_act) in out_acts_arr.iter().enumerate()
      {
         let psi = deriv_wrt_out[out_it] * threshold_func_prime(*out_act);

         for (in_it, dest_wrt_inp) in dest_deriv_wrt_inp.iter_mut().enumerate()
         {
            *dest_wrt_inp += psi * self.get_weight(in_it, out_it);

            let deriv_err_wrt_weight = -inp_acts_arr[in_it] * psi;
            *self.get_delta_weight_mut(in_it, out_it) = -learn_rate * deriv_err_wrt_weight;
         }
      } // for (out_it, act) in outp.iter().enumerate()
   } // pub fn feed_backwards(...)

   /// Changes the weights throughout the network according to the stored deltas,
   /// and then clears the deltas to zero for the next backpropagation step.
   pub fn apply_delta_weights(&mut self)
   {
      for (weight, delta) in self.weights.iter_mut().zip(self.delta_weights.iter())
      {
         *weight += delta;
      }

      self.delta_weights.fill(0.0f64);
   }
} // impl NetworkLayer

impl NeuralNetwork
{
   /**
    * Allocates memory to store a NeuralNetwork. All fields
    * are populated with dummy values, and the Network is NOT initialized,
    * contrary to RAII philosophy. See `set_and_echo_config` in `config.rs`
    * for the code that actually handles network initialization.
    */
   pub fn new() -> NeuralNetwork
   {
      NeuralNetwork {
         layers: Box::new([]),
         activations: Box::new([]),
         threshold_func: ident,
         threshold_func_deriv: ident_deriv,
         learn_rate: 1 as NumT,
         derivs: [Box::new([]), Box::new([])],

         max_iterations: 0,
         error_cutoff: 0 as NumT,
         do_training: false,
         printout_period: 0,
      } // NeuralNetwork
   } // pub fn new() -> NeuralNetwork

   /// Gets a mutable reference to the input array of the network.
   /// `feed_forward` reads its input from here.
   pub fn get_inputs(&mut self) -> &mut [NumT]
   {
      &mut self.activations[0]
   }

   /// Gets the array of the outputs of the network.
   /// `feed_forward` will write its output here.
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
    *
    * This function returns the value of the cost function on this particular
    * training case. The cost is defined as 0.5 * (expected value - actual output)^2
    */
   pub fn feed_backwards(&mut self, expected_out: &[NumT]) -> NumT
   {
      assert_eq!(expected_out.len(), self.get_outputs().len());

      let diff = expected_out[0] - self.get_outputs()[0];
      let mut error = 0.5 * diff * diff;
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

      error
   } // pub fn feed_backwards(&mut self, expected_out: &[NumT]) -> NumT

   /// Applies the changes stored in `delta_weights` for each of the layers in the network.
   /// See documentation for `NetworkLayer::apply_delta_weights` above.
   pub fn apply_delta_weights(&mut self)
   {
      for layer in self.layers.iter_mut()
      {
         layer.apply_delta_weights();
      }
   }
} // impl NeuralNetwork
