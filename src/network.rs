/**
 * network.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Defines the NeuralNetwork struct and the methods to
 * run the network and perform backpropagation learning.
 *
 * The `NetworkLayer` and `NeuralNetwork` structs implement the functions
 * `feed_forward` to run on an input, `feed_backward` to perform backpropagation,
 * and `apply_delta_weights` to run backpropagation by applying
 * the changes calculated in backpropagation.
 *
 * This file also defines the `Datapoint` struct for testing and training data.
 */
use crate::config::{ident, ident_deriv, FuncT, NumT};

/**
 * These are array indices into `NeuralNetwork::activations`,
 * specifying where to read from and where to write to during the
 * backpropagation step. See `NeuralNetwork::feed_backward` and `NetworkLayer::feed_backward`
 */
const INPUT_DERIV: usize = 0;
const OUTPUT_DERIV: usize = 1;

/// A datapoint in the dataset with an input value and
/// (optionally for training) the expected output. Used for test and train data.
#[derive(Debug)]
pub struct Datapoint
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
    *
    * In the design, elements in this array correspond to the a_k values, the h_j values,
    * and the F_i values.
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
   pub omegas: [Box<[NumT]>; 2],

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
   /// The weights and delta_weights matrices are flattened stored in row-major order.
   pub weights: Box<[NumT]>,
   pub thetas_out: Box<[NumT]>,

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
      let cond_allocate = |size: usize| {
         if do_training
         {
            vec![0.0; size].into_boxed_slice()
         }
         else
         {
            Box::new([])
         }
      };

      NetworkLayer
      {
         num_inputs,
         num_outputs,
         weights: vec![0.0; (num_inputs * num_outputs) as usize].into_boxed_slice(),
         thetas_out: cond_allocate(num_outputs as usize)
      }
   } // fn new(num_inputs: i32, num_outputs: i32, do_training: bool) -> NetworkLayer

   /**
    * These functions are used for indexing into the weight matrix.
    * These make code easier since I used flattened weight arrays. Instead of using a 2d
    * array to store the weight matrices, I used 1d arrays in row-major order.
    *
    * Both functions have the following preconditions:
    * 0 <= `inp_index` < `self.num_inputs` and 0 <= `out_index` < `self.num_outputs`.
    */
   pub fn get_weight(&self, inp_index: usize, out_index: usize) -> &NumT
   {
      debug_assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      &self.weights[inp_index * self.num_outputs as usize + out_index]
   }

   pub fn get_weight_mut(&mut self, inp_index: usize, out_index: usize) -> &mut NumT
   {
      debug_assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      &mut self.weights[inp_index * self.num_outputs as usize + out_index]
   }

   /**
    * Feeds the `inp_act_arr` input activation array forward through this network layer,
    * using the `threshold_func` activation function.
    * Outputs the results into the `dest_h_out` array.
    *
    * Precondition: `inp_act_arr` must have length equal to `self.num_inputs`
    * and `dest_h_out` must have length equal to `self.num_outputs`.
    *
    * The generic template variable `WRITE_THETA` controls whether the
    * intermediate theta values are saved. It should only be true for
    * training, as the theta arrays are not allocated when `do_training` is false.
    */
   pub fn feed_forward<const WRITE_THETA: bool>(&mut self, inp_act_arr: &[NumT],
                                                dest_h_out_arr: &mut [NumT],
                                                threshold_func: FuncT)
   {
      debug_assert_eq!(inp_act_arr.len(), self.num_inputs as usize);
      debug_assert_eq!(dest_h_out_arr.len(), self.num_outputs as usize);

      for out_it in 0..self.num_outputs as usize
      {
         let theta = (0..self.num_inputs as usize)
            .map(|in_it| {
               self.get_weight(in_it, out_it) * inp_act_arr[in_it]
            }).sum::<NumT>();

         dest_h_out_arr[out_it] = threshold_func(theta);

         // this is NOT a conditional checked at runtime.
         // this is evaluated at compile time as WRITE_THETA is a template variable.
         if WRITE_THETA
         {
            self.thetas_out[out_it] = theta;
         }
      } // for out_it in 0..self.num_outputs as usize
   } // pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)

   /**
    * Performs backpropagation on this network layer and takes a step
    * in parameter space (i.e. steps the weights along the gradients to minimize cost).
    *
    * `inp_acts_arr` is an array of the original inputs fed into this layer.
    * In the design, it corresponds to the `a` vector for the hidden layer
    * or the `h` vector for the output layer. Its length MUST be equal to `self.num_inputs`.
    *
    * `learn_rate` is the lambda value, the step size for backpropagation.
    *
    * `threshold_func_prime` is the derivative of the threshold function that was used
    * in this layer.
    *
    * `prev_omegas` is the derivative of the cost function with respect to
    * each of this layer's outputs. In the design, it corresponds to the lowercase omega
    * for the output layer, and the uppercase omega for the hidden layer.
    * Its length MUST be equal to `self.num_outputs`
    *
    * `dest_next_omegas` is an array to output into, where this function will write
    * the derivative of the cost function with respect to each of this layer's inputs.
    * In the design, it corresponds to the capital omega vector for the output layer.
    * Its length MUST be equal to `self.num_inputs`.
    * This value will then be fed into the `feed_backward` function of the previous layer.
    * This layer's input is the previous layer's output, so this layer's `dest_next_omegas`
    * becomes the new `prev_omegas` of the previous layer.
    */
   pub fn feed_backward(&mut self, inp_acts_arr: &[NumT], learn_rate: NumT,
                        threshold_func_prime: FuncT, prev_omegas: &[NumT],
                        dest_next_omegas: &mut [NumT])
   {
      debug_assert_eq!(inp_acts_arr.len(), self.num_inputs as usize);
      debug_assert_eq!(prev_omegas.len(), self.num_outputs as usize);
      debug_assert_eq!(dest_next_omegas.len(), self.num_inputs as usize);

      dest_next_omegas.fill(0.0);
      for out_it in 0..self.num_outputs as usize
      {
         let psi = prev_omegas[out_it] * threshold_func_prime(self.thetas_out[out_it]);

         debug_assert!(psi.is_finite() && !psi.is_nan());
         for (in_it, dest_wrt_inp) in dest_next_omegas.iter_mut().enumerate()
         {
            *dest_wrt_inp += psi * self.get_weight(in_it, out_it);
            *self.get_weight_mut(in_it, out_it) += learn_rate * inp_acts_arr[in_it] * psi;
         }
      } // for out_it in 0..self.num_outputs as usize
   } // pub fn feed_backward(...)
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
      NeuralNetwork
      {
         layers: Box::new([]),
         activations: Box::new([]),
         threshold_func: ident,
         threshold_func_deriv: ident_deriv,
         learn_rate: 1.0,
         omegas: [Box::new([]), Box::new([])],
         max_iterations: 0,
         error_cutoff: 0.0,
         do_training: false,
         printout_period: 0
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
    *
    * The generic template variable `WRITE_THETA` controls whether the
    * intermediate theta values are saved for backpropagation. It should only be true for
    * training, as the theta arrays are not allocated when `do_training` is false.
    */
   pub fn feed_forward<const WRITE_THETA: bool>(&mut self)
   {
      for (index, layer) in self.layers.iter_mut().enumerate()
      {
         // hack to appease the borrow checker. see note in `NeuralNetwork::feed_forward`.
         let (input_slice, output_slice) = self.activations.split_at_mut(index + 1);

         let input_arr = &input_slice[index];
         let dest_output_arr = &mut output_slice[0];
         layer.feed_forward::<WRITE_THETA>(input_arr, dest_output_arr, self.threshold_func);
      }
   } // pub fn feed_forward(&mut self)

   /**
    * Calculates the error and fills
    * the last layer's derivative field in preparation for backprop.
    *
    * The length of `target_out` MUST be equal to the size of this network's output.
    * This function returns the value of the error function on this particular
    * training case. The error is defined as the sum of 0.5 * (target value - actual output)^2
    * over all the output nodes.
    */
   pub fn calculate_error(&mut self, target_out: &[NumT]) -> NumT
   {
      debug_assert_eq!(target_out.len(), self.get_outputs().len());

      let mut error = 0.0;
      for i in 0..self.get_outputs().len()
      {
         let little_omega = target_out[i] - self.get_outputs()[i];
         error += 0.5 * little_omega * little_omega;
         self.omegas[INPUT_DERIV][i] = little_omega;
      }

      error
   }

   /**
    * Runs backpropagation through the full neural network.
    * The `target_out` is the expected output array of the network and is
    * used to compute the cost function and gradient. Its length must
    * be equal to this network's output size.
    *
    * This function returns the error value for this training case.
    * See `NeuralNetwork::calculate_error`
    */
   pub fn feed_backward(&mut self, target_out: &[NumT]) -> NumT
   {
      let error = self.calculate_error(target_out);

      for (index, layer) in self.layers.iter_mut().enumerate().rev()
      {
         // rust cannot let us borrow 2 values from the same array at the same time,
         // so we have to use this hack to appease the borrow checker.
         // The code becomes very ugly since these functions return slices.
         let (inp_deriv_slice, outp_deriv_slice) = self.omegas.split_at_mut(1);

         layer.feed_backward(&self.activations[index],
                             self.learn_rate,
                             self.threshold_func_deriv,
                             &inp_deriv_slice[0][..layer.num_outputs as usize],
                             &mut outp_deriv_slice[0][..layer.num_inputs as usize]);

         // the derivatives outputted by this layer become the derivatives
         // inputted into the previous layer. This layer's input derivatives
         // will become overwritten.
         self.omegas.swap(INPUT_DERIV, OUTPUT_DERIV);
      } // for (index, layer) in self.layers.iter_mut().enumerate().rev()

      error
   } // pub fn feed_backward(&mut self, expected_out: &[NumT]) -> NumT
} // impl NeuralNetwork
