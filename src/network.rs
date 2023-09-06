use crate::config::ConfigValue::{IntList, Text};
use crate::config::{ident, ident_deriv, make_err, ConfigValue, FuncT, NumT};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug)]
pub struct NeuralNetwork
{
   pub layers: Box<[NetworkLayer]>,
   pub activations: Box<[Box<[NumT]>]>,

   pub threshold_func: FuncT,
   pub threshold_func_deriv: FuncT,
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
      }
   }

   pub(crate) fn get_outputs(&self) -> &[NumT]
   {
      return &self.activations[self.activations.len() - 1];
   }

   pub(crate) fn feed_forward(&mut self, input: &[NumT])
   {
      self.layers[0].feed_forward(input, &mut self.activations[0], self.threshold_func);
      for (layer, index) in self.layers.iter().skip(1).zip(0..)
      {
         let (input_slice, output_slice) = self.activations.split_at_mut(index + 1);
         layer.feed_forward(
            &input_slice[index],
            &mut output_slice[0],
            self.threshold_func,
         );
      }
   }
} // impl NeuralNetwork
