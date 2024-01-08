use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;
use rand_distr::StandardNormal;
/**
 * network.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Defines the NeuralNetwork struct and the methods to
 * run the network and perform backpropagation learning.
 *
 * Table of contents:
 * `Datapoint`: a container for testing and training data
 *
 * `NetworkLayer`
 *    `pub fn new(num_inputs: i32, num_outputs: i32) -> NetworkLayer`
 *    ```pub fn feed_forward(&mut self, inp_act_arr: &[NumT],
 *                           dest_h_out_arr: &mut [NumT],
 *                           threshold_func: FuncT)```
 *    ```pub fn feed_backward(&mut self, inp_acts_arr: &[NumT], learn_rate: NumT,
 *                          threshold_func_prime: FuncT, prev_omegas: &[NumT],
 *                          dest_next_omegas: &mut [NumT])```
 *
 * `NeuralNetwork`
 *    `pub fn new() -> NeuralNetwork`
 *    `pub fn get_inputs(&mut self) -> &mut [NumT]`
 *    `pub fn get_outputs(&self) -> &[NumT]`
 *    `pub fn feed_forward(&mut self)`
 *    `pub fn calculate_error(&mut self, target_out: &[NumT]) -> NumT`
 *    `pub fn feed_backward(&mut self, target_out: &[NumT]) -> NumT`
 */
use crate::config::{ident, ident_deriv, FuncT, NumT};

/**
 * These are array indices into `NeuralNetwork::activations`,
 * specifying where to read from and where to write to during the
 * backpropagation step. See `NeuralNetwork::feed_backward` and `NetworkLayer::feed_backward`
 */
const INPUT_DERIV: usize = 0;
const OUTPUT_DERIV: usize = 1;

/**
 * A datapoint in the dataset with an input value and
 * (optionally for training) the expected output. Used for test and train data.
 */
#[derive(Debug)]
pub struct Datapoint
{
   pub inputs: Box<[NumT]>,
   pub expected_outputs: Box<[NumT]>,
}

#[derive(Debug)]
pub struct TrainParams
{
   pub learn_rate: NumT,
   pub dropout_prob: NumT,
   pub beta1: NumT,
   pub beta2: NumT,
   pub eps: NumT,
   pub weight_decay: NumT,

   /// The activation functions apply element-wise to each layer's output.
   pub threshold_func: FuncT,
   pub threshold_func_deriv: FuncT,
}

/// Neural network, containing both layer weights and configuration data
#[derive(Debug)]
pub struct NeuralNetwork
{
   /**
    * Stores the weight matrix of each layer, along with the layer's dimensions.
    * When training, the layer's outputs before applying the activation function are also stored.
    * See `NetworkLayer` for more information.
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
   pub dropouts: Box<[Box<[bool]>]>,

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

   /**
    * The following values are copied directly from
    * the configuration file and have the same names.
    * For more information, see the specification of the configuration format
    * in the documentation for `set_and_echo_config` in `config.rs`.
    */
   pub train_params: TrainParams,
   pub error_cutoff: NumT,
   pub max_iterations: i32,
   pub printout_period: i32,
   pub batch_size: i32,
   pub checkpoint_period: i32,
   pub do_training: bool,
} // pub struct NeuralNetwork

/**
 * Represents an individual hidden (or output) layer in the network.
 * Stores the weight array.
 */
#[derive(Debug)]
pub struct NetworkLayer
{
   /// The weight matrix is flattened stored in row-major order.
   pub weights: Box<[NumT]>,
   pub delta_weights: Box<[NumT]>,

   /// Parameters for the ADAM optimizer
   pub momentum: Box<[NumT]>,
   pub velocity: Box<[NumT]>,

   pub num_inputs: i32,
   pub num_outputs: i32,
} // struct NetworkLayer

impl NetworkLayer
{
   /**
    * Allocates memory for a neural network layer according to the dimensions
    * of the inputs and outputs specified by `num_inputs` and `num_outputs`.
    *
    * This function fills the fields of the network with dummy values.
    * See `set_and_echo_config` in `config.rs` for the code to initialize the network.
    */
   pub fn new(num_inputs: i32, num_outputs: i32) -> NetworkLayer
   {
      let vec = vec![0.0; (num_inputs * num_outputs) as usize];
      NetworkLayer
      {
         num_inputs,
         num_outputs,
         weights: vec.clone().into_boxed_slice(),
         momentum: vec.clone().into_boxed_slice(),
         delta_weights: vec.clone().into_boxed_slice(),
         velocity: vec.into_boxed_slice()
      }
      // NetworkLayer
   } // fn new(num_inputs: i32, num_outputs: i32, do_training: bool) -> NetworkLayer

   /**
    * This functions is used for indexing into the weight, momentum, and velocity matrices.
    * These make code easier since I used flattened weight arrays. Instead of using a 2d
    * array to store the weight matrices, I used 1d arrays in row-major order.
    *
    * Precondition:
    * 0 <= `inp_index` < `self.num_inputs` and 0 <= `out_index` < `self.num_outputs`.
    */
   pub fn coord_to_idx(&self, inp_index: usize, out_index: usize) -> usize
   {
      debug_assert!(inp_index < self.num_inputs as usize && out_index < self.num_outputs as usize);
      inp_index * self.num_outputs as usize + out_index
   }

   /**
    * Feeds the `inp_act_arr` input activation array forward through this network layer,
    * using the `threshold_func` activation function.
    * Outputs the results into the `dest_h_out` array.
    *
    * Precondition: `inp_act_arr` must have length equal to `self.num_inputs`
    * and `dest_h_out` must have length equal to `self.num_outputs`.
    */
   pub fn feed_forward(&mut self, inp_act_arr: &[NumT], dest_h_out_arr: &mut [NumT],
                       threshold_func: FuncT, activ: bool)
   {
      debug_assert_eq!(inp_act_arr.len(), self.num_inputs as usize);
      debug_assert_eq!(dest_h_out_arr.len(), self.num_outputs as usize);

      for out_it in 0..self.num_outputs as usize
      {
         let theta = (0..self.num_inputs as usize)
            .map(|in_it| self.weights[self.coord_to_idx(in_it, out_it)] * inp_act_arr[in_it])
            .sum::<NumT>();

         dest_h_out_arr[out_it] = if activ { threshold_func(theta) } else { theta };
      } // for out_it in 0..self.num_outputs as usize
   } // pub fn feed_forward<...>(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)

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
   pub fn feed_backward(&mut self, inp_acts_arr: &[NumT], out_acts_arr: &[NumT],
                        params: &TrainParams,
                        dropped_outputs: &[bool],
                        prev_omegas: &[NumT], dest_next_omegas: &mut [NumT],
                        step: i32, activ: bool)
   {
      debug_assert_eq!(inp_acts_arr.len(), self.num_inputs as usize);
      debug_assert_eq!(prev_omegas.len(), self.num_outputs as usize);
      debug_assert_eq!(dest_next_omegas.len(), self.num_inputs as usize);
      debug_assert_eq!(out_acts_arr.len(), self.num_outputs as usize);
      debug_assert_eq!(dropped_outputs.len(), self.num_outputs as usize);

      for (in_it, dest_wrt_inp) in dest_next_omegas.iter_mut().enumerate()
      {
         *dest_wrt_inp = (0..self.num_outputs as usize).map(|out_it|
         {
            let psi = if dropped_outputs[out_it]
            {
               0.0
            }
            else
            {
               prev_omegas[out_it] * if activ {
                  (params.threshold_func_deriv)(out_acts_arr[out_it])
               } else {
                  1.0
               }
            };

            debug_assert!(psi.is_finite() && !psi.is_nan());

            let idx = self.coord_to_idx(in_it, out_it);
            let g = inp_acts_arr[in_it] * psi + params.weight_decay * self.weights[idx];

            self.momentum[idx] *= params.beta1;
            self.momentum[idx] += (1.0 - params.beta1) * g;

            self.velocity[idx] *= params.beta2;
            self.velocity[idx] += (1.0 - params.beta2) * g * g;

            // sqrt(1 - B2^t) / (1 - B1^t) can cause a div by zero error if t is small
            // so make sure that step > 0
            let a = params.learn_rate * (1.0 - params.beta2.powi(step)).sqrt()
               / (1.0 - params.beta1.powi(step));

            self.delta_weights[idx] += -a * self.momentum[idx] / (self.velocity[idx].sqrt() + params.eps)
                                       - params.weight_decay * self.weights[idx];

            psi * self.weights[idx]
         }).sum(); // (0..self.num_inputs as usize).map(...).sum()
      } // for (in_it, dest_wrt_inp) in dest_next_omegas.iter_mut().enumerate()
   } // pub fn feed_backward(...)

   pub fn apply_deltas(&mut self)
   {
      for (idx, w) in self.weights.iter_mut().enumerate()
      {
         *w += self.delta_weights[idx];
      }
      self.delta_weights.fill(0.0);
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
      NeuralNetwork
      {
         layers: Box::new([]),
         activations: Box::new([]),
         omegas: [Box::new([]), Box::new([])],
         max_iterations: 0,
         error_cutoff: 0.0,
         do_training: false,
         printout_period: 0,
         checkpoint_period: 0,
         batch_size: 0,
         dropouts: Box::new([]),
         train_params: TrainParams {
            threshold_func: ident,
            threshold_func_deriv: ident_deriv,
            learn_rate: 1.0,
            dropout_prob: 0.0,
            beta2: 0.0,
            beta1: 0.0,
            eps: 0.0,
            weight_decay: 0.0
         },
      } // NeuralNetwork
   } // pub fn new() -> NeuralNetwork

   pub fn add_noise(&mut self, rng: &mut ThreadRng)
   {
      for inp in self.get_inputs().iter_mut()
      {
         *inp += 0.2 * rng.sample::<f32,_>(StandardNormal);
         if rng.gen_bool(0.1)
         {
            *inp = 0.0;
         }
      }
   }

   /**
    * Gets a mutable reference to the input array of the network.
    * `feed_forward` reads its input from here.
    */
   pub fn get_inputs(&mut self) -> &mut [NumT]
   {
      &mut self.activations[0]
   }

   /**
    * Gets the array of the outputs of the network.
    * `feed_forward` will write its output here.
    */
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
      let layer_len = self.layers.len();
      for (index, layer) in self.layers.iter_mut().enumerate()
      {
         /*
          * rust's borrow checking will not allow us to borrow 2 mutable values from the
          * same slice at the same time, so we must split the slice into 2 halves first.
          */
         let (input_slice, output_slice) = self.activations.split_at_mut(index + 1);

         let input_arr = &input_slice[index];
         let dest_output_arr = &mut output_slice[0];
         layer.feed_forward(input_arr, dest_output_arr,
                            self.train_params.threshold_func, index != layer_len - 1);

         if index < layer_len - 1
         {
            for (drop_idx, drop) in self.dropouts[index + 1].iter().enumerate()
            {
               if *drop
               {
                  dest_output_arr[drop_idx] = 0.0;
               }
            }
         }
      }

      // apply softmax
      let mut sum = 0.0;
      for i in self.activations.last_mut().unwrap().iter_mut()
      {
         *i = i.exp();
         sum += *i;
      }

      for i in self.activations.last_mut().unwrap().iter_mut()
      {
         *i /= sum;
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
      let targ_sum = target_out.iter().sum::<NumT>();
      for i in 0..self.get_outputs().len()
      {
         let little_omega = self.get_outputs()[i] * targ_sum - target_out[i];
         self.omegas[INPUT_DERIV][i] = little_omega;
         error += -target_out[i] * self.get_outputs()[i].ln();
      }

      error / self.get_outputs().len() as NumT
   } // fn calculate_error(&mut self, target_out: &[NumT]) -> NumT

   /**
    * Runs backpropagation through the full neural network.
    * The `target_out` is the expected output array of the network and is
    * used to compute the cost function and gradient. Its length must
    * be equal to this network's output size.
    *
    * This function returns the error value for this training case.
    * See `NeuralNetwork::calculate_error`
    */
   pub fn feed_backward(&mut self, target_out: &[NumT], step: i32) -> NumT
   {
      let error = self.calculate_error(target_out);

      let sz = self.layers.len() - 1;
      for (index, layer) in self.layers.iter_mut().enumerate().rev()
      {
         let (inp_deriv_slice, outp_deriv_slice) = self.omegas.split_at_mut(1);

         layer.feed_backward(&self.activations[index],
                             &self.activations[index + 1],
                             &self.train_params,
                             &self.dropouts[index + 1],
                             &inp_deriv_slice[0][..layer.num_outputs as usize],
                             &mut outp_deriv_slice[0][..layer.num_inputs as usize],
                             step, index != sz);

         /*
          * the derivatives outputted by this layer become the derivatives
          * inputted into the previous layer. This layer's input derivatives
          * will become overwritten by the next layer's output derivatives on the next iteration.
          */
         self.omegas.swap(INPUT_DERIV, OUTPUT_DERIV);
      } // for (index, layer) in self.layers.iter_mut().enumerate().rev()

      error
   } // pub fn feed_backward(&mut self, target_out: &[NumT]) -> NumT

   pub fn randomize_dropouts(&mut self)
   {
      let mut rng = thread_rng();
      let size = self.dropouts.len() - 1; // do not perform dropout on outputs
      for layer in self.dropouts.iter_mut().take(size)
      {
         for v in layer.iter_mut()
         {
            *v = rng.gen_bool(self.train_params.dropout_prob as f64);
         }
      }
   }

   pub fn zero_dropouts(&mut self)
   {
      for layer in self.dropouts.iter_mut()
      {
         layer.fill(false);
      }
   }

   pub fn apply_deltas(&mut self)
   {
      for layer in self.layers.iter_mut()
      {
         layer.apply_deltas();
      }
   }
} // impl NeuralNetwork
