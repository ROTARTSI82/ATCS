/**
 * serialize.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Contains two utility functions for serializing and deserializing
 * full NeuralNetwork weights to a binary file.
 *
 * Table of contents:
 * `fn write_net_to_file(net: &NeuralNetwork, filename: &str) -> Result<(), std::io::Error>`
 * `fn consume_i32(list: &[u8]) -> Result<i32, std::io::Error>`
 * `fn consume_num(list: &[u8]) -> Result<NumT, std::io::Error>`
 * `fn read_net_from_file(net: &mut NeuralNetwork, filename: &str) -> Result<(), std::io::Error>`
 */
use crate::config::{make_err, NumT};
use crate::network::NeuralNetwork;

use std::fs::File;
use std::io::{Read, Write};

const I32_SIZE: usize = std::mem::size_of::<i32>();
const NUM_SIZE: usize = std::mem::size_of::<NumT>();
const MAGIC_FILE_HEADER: &[u8] = b"ATNeuralNetwork";

/**
 * Saves the NeuralNetwork `net` to a file at `filename` in a binary format.
 * On success, returns Ok(()). If there is an error with the file I/O,
 * an Err(io::Error) is returned with the error that occurred.
 *
 * Here is the packed binary format:
 * 1. The ASCII bytes b"ATNeuralNetwork" (specified in MAGIC_FILE_HEADER)
 * 2. The number of layers in the network.  (32-bit big endian integer)
 * 3. The number of inputs expected by the network. (32-bit big endian integer)
 * 4. The number of outputs for each layer successively, including the number of outputs
 *    as the last value. (This field is an array of 32-bit big endian integers)
 *    The number of values in this field is equal to the value of field #2.
 * 5. The weights of each layer successively. The format is the same as the format
 *    of the `NetworkLayer::weights` field (a packed matrix in row-major order)
 *    This field is an array of `NumT` values as converted to binary by rust's `to_be_bytes`
 */
pub fn write_net_to_file(net: &NeuralNetwork, filename: &str) -> Result<(), std::io::Error>
{
   let mut bytes = MAGIC_FILE_HEADER.to_vec();
   bytes.extend((net.layers.len() as u32).to_be_bytes());

   let input_size = net.layers
                       .first()
                       .ok_or(make_err("layers empty"))?
                       .num_inputs
                       .to_be_bytes();
   let layer_bytes = net.layers
                        .iter()
                        .flat_map(|it| it.num_outputs.to_be_bytes());

   bytes.extend(input_size);
   bytes.extend(layer_bytes);

   for layer in net.layers.iter()
   {
      bytes.extend(layer.weights.iter().flat_map(|it| it.to_be_bytes()));
   }

   let mut file = File::create(filename)?;
   file.write_all(bytes.as_slice())?;

   println!("saved neural network to file `{}`", filename);
   Ok(())
} // pub fn write_net_to_file(net: &NeuralNetwork, filename: &str) -> Result<(), io::Error>

/**
 * These functions take an array of u8 bytes and interprets the first bytes
 * as either a NumT or an i32, returning it as a Result. If the u8 array
 * does not have enough values to create the output type, the Err() variant
 * is returned with a message warning about the corrupt data.
 *
 * rust doesn't have an easy way to do this with generics, since you can't
 * call from_be_bytes() with an unknown type. In order to do that, I would
 * have to write a trait and implement it for both i32 and NumT.
 * At this point, it's easier to just copy and paste these methods twice.
 */
fn consume_i32(list: &[u8]) -> Result<i32, std::io::Error>
{
   let bytes = list[..I32_SIZE].try_into()
                               .map_err(|_| make_err("corrupt network depth/input size"))?;

   Ok(i32::from_be_bytes(bytes))
}

fn consume_num(list: &[u8]) -> Result<NumT, std::io::Error>
{
   let bytes = list[..NUM_SIZE].try_into()
                               .map_err(|_| make_err("corrupt network weight"))?;

   Ok(NumT::from_be_bytes(bytes))
}

/**
 * Loads a NeuralNetwork from a file. Any file I/O errors or
 * corrupted/mismatched data between the binary file and the configuration
 * will result in this function returning an Err(io::Error) variant with
 * a message. Otherwise, all data is loaded into the network and Ok(()) is returned.
 *
 * Refer to the documentation for `write_net_to_file` for a specification of the binary format.
 */
pub fn read_net_from_file(net: &mut NeuralNetwork, filename: &str) -> Result<(), std::io::Error>
{
   let mut bytes = Vec::new();
   {
      let mut file = File::open(filename)?;
      file.read_to_end(&mut bytes)?;
   }

   if !bytes.starts_with(MAGIC_FILE_HEADER)
   {
      Err(make_err(format!("magic header missing from {}", filename).as_str()))?;
   }

   let bytes = &bytes[MAGIC_FILE_HEADER.len()..];

   let model_layers = consume_i32(bytes)?;
   let config_layers = net.layers.len() as i32;
   if model_layers != config_layers
   {
      Err(make_err(format!("model file is {} layers, config is {} layers",
                           model_layers, config_layers).as_str()))?;
   }

   let bytes = &bytes[I32_SIZE..];
   let model_input = consume_i32(bytes)?;
   let config_input = net.layers
                         .first()
                         .ok_or(make_err("layers empty"))?
                         .num_inputs;

   if model_input != config_input
   {
      Err(make_err(format!("model file is {} inputs, config is {} inputs",
                           model_input, config_input).as_str()))?;
   }

   let bytes = &bytes[I32_SIZE..];

   for (it, layer) in net.layers.iter().enumerate()
   {
      let model_out = i32::from_be_bytes(
         bytes[it * I32_SIZE..it * I32_SIZE + I32_SIZE]
            .try_into()
            .map_err(|_| make_err("corrupt layer size"))?,
      );

      if model_out != layer.num_outputs
      {
         Err(make_err(format!("model file layer {} is {} outputs, config is {} outputs",
                              it, model_out, layer.num_outputs).as_str()))?;
      }
   } // for (it, layer) in net.layers().iter().enumerate()

   let bytes = &bytes[I32_SIZE * net.layers.len()..];

   let mut ptr = 0;
   for layer in net.layers.iter_mut()
   {
      for weight in layer.weights.iter_mut()
      {
         *weight = consume_num(&bytes[NUM_SIZE * ptr..])?;
         ptr += 1;
      }
   }

   println!("loaded neural network from file `{}`", filename);
   Ok(())
} // pub fn load_net_from_file(net: &mut NeuralNetwork, filename: &str) -> Result<(), io::Error>
