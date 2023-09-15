use crate::config::{ident, ident_deriv, FuncT, NumT};

#[derive(Debug)]
pub struct TrainCase
{
   pub inp: Box<[NumT]>,
   pub outp: Box<[NumT]>,
}

#[derive(Debug)]
pub struct NeuralNetwork
{
   pub layers: Box<[NetworkLayer]>,
   pub activations: Box<[Box<[NumT]>]>,
   pub derivs: [Box<[NumT]>; 2],

   pub threshold_func: FuncT,
   pub threshold_func_deriv: FuncT,

   pub learn_rate: NumT,
   pub max_width: i32,

   pub max_iters: i32,
   pub err_threshold: NumT,
   pub do_training: bool,

   pub printout_period: i32,
}

#[derive(Debug)]
pub struct NetworkLayer
{
   pub weights: Box<[NumT]>,
   pub biases: Box<[NumT]>,

   pub num_inputs: i32,
   pub num_outputs: i32,
}

impl NetworkLayer
{
   pub fn new(num_inputs: i32, num_outputs: i32) -> NetworkLayer
   {
      NetworkLayer {
         num_inputs,
         num_outputs,
         weights: vec![0 as NumT; (num_inputs * num_outputs) as usize].into_boxed_slice(),
         biases: vec![0 as NumT; num_outputs as usize].into_boxed_slice(),
      }
   }

   pub fn get_weight_mut(&mut self, row: i32, col: i32) -> &mut NumT
   {
      assert!(
         row < self.num_inputs && col < self.num_outputs,
         "out of bounds access to weights"
      );
      &mut self.weights[(row * self.num_outputs + col) as usize]
   }

   pub fn get_weight(&self, row: i32, col: i32) -> &NumT
   {
      assert!(
         row < self.num_inputs && col < self.num_outputs,
         "out of bounds access to weights"
      );
      &self.weights[(row * self.num_outputs + col) as usize]
   }

   pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)
   {
      assert!(
         inp.len() == self.num_inputs as usize && out.len() == self.num_outputs as usize,
         "unexpected in/out dimensions for neural network layer: expected ({}, {}) got ({}, {})",
         self.num_inputs,
         self.num_outputs,
         inp.len(),
         out.len()
      );

      for out_it in 0..self.num_outputs
      {
         let product_part = (0..self.num_inputs)
            .map(|in_it| self.get_weight(in_it, out_it) * inp[in_it as usize])
            .sum::<NumT>();

         out[out_it as usize] = act(product_part + self.biases[out_it as usize]);
      }
   }

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

      const MAGN_FACTOR: NumT = 0 as NumT;

      dest_deriv_wrt_inp.fill(0 as NumT);
      for (out_it, act) in outp.iter().enumerate()
      {
         // dCost/dInner = dCost/dOutput * dOutput/dInner
         let deriv_wrt_inner = deriv_wrt_outp[out_it] * act_prime(*act);

         for (in_it, wrt_outer) in dest_deriv_wrt_inp.iter_mut().enumerate()
         {
            *wrt_outer += deriv_wrt_inner * self.get_weight(in_it as i32, out_it as i32);

            *self.get_weight_mut(in_it as i32, out_it as i32) +=
               deriv_wrt_inner * inp[in_it] * learn_rate
                  - MAGN_FACTOR * self.get_weight(in_it as i32, out_it as i32);
         }

         self.biases[out_it] += deriv_wrt_inner * learn_rate - MAGN_FACTOR * self.biases[out_it];
         // self.biases[out_it] = 0 as NumT;
      }
   }
}

impl NeuralNetwork
{
   pub(crate) fn new() -> NeuralNetwork
   {
      NeuralNetwork {
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
      }
   }

   pub(crate) fn get_outputs(&self) -> &[NumT]
   {
      &self.activations[self.activations.len() - 1]
   }

   pub(crate) fn feed_forward(&mut self)
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
   }

   pub fn get_inputs(&mut self) -> &mut [NumT]
   {
      &mut self.activations[0]
   }

   // returns the loss function for this test case
   pub fn feed_backwards(&mut self, expected_out: &[NumT]) -> NumT
   {
      assert_eq!(expected_out.len(), self.get_outputs().len());

      let mut cost = 0 as NumT;
      // println!("got {:?} expected {:?}", self.get_outputs(), expected_out);
      for (idx, expected) in expected_out.iter().enumerate()
      {
         // i've messed up a minus sign somewhere, this somehow is what i need to do??
         let diff = expected - self.get_outputs()[idx];
         cost += 0.5 * diff * diff;
         self.derivs[0][idx] = diff;
      }

      for (index, layer) in self.layers.iter_mut().enumerate().rev()
      {
         let (first, second) = self.derivs.split_at_mut(1);
         let (inp_slice, outp_slice) = self.activations.split_at(index + 1);
         layer.feed_backwards(
            &inp_slice[index],
            &outp_slice[0],
            self.learn_rate,
            self.threshold_func_deriv,
            &first[0][..layer.num_outputs as usize],
            &mut second[0][..layer.num_inputs as usize],
         );
         self.derivs.swap(0, 1);
      }

      cost
   }
} // impl NeuralNetwork
