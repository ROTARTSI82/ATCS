use crate::config::{make_err, NumT};
use crate::network::NeuralNetwork;
use std::fs::File;
use std::io::{Read, Write};

const MAGIC_HEADER: &'static [u8] = b"ATNeuralNetwork";

pub fn write_net_to_file(net: &NeuralNetwork, filename: &str) -> Result<(), std::io::Error>
{
   let mut bytes = MAGIC_HEADER.to_vec();
   bytes.extend((net.layers.len() as u32).to_be_bytes());

   let input_size = net
      .layers
      .first()
      .ok_or(make_err("layers empty"))?
      .num_inputs
      .to_be_bytes();
   let layer_bytes = net
      .layers
      .iter()
      .map(|it| it.num_outputs.to_be_bytes())
      .flatten();

   bytes.extend(input_size);
   bytes.extend(layer_bytes);

   let flatten_to_bytes = |list: &Box<[NumT]>| {
      list
         .to_vec()
         .into_iter()
         .map(|it| it.to_be_bytes())
         .flatten()
   };

   for layer in net.layers.iter()
   {
      bytes.extend(flatten_to_bytes(&layer.weights));
      bytes.extend(flatten_to_bytes(&layer.biases));
   }

   let mut file = File::create(filename)?;
   file.write_all(bytes.as_slice())?;

   println!("saved neural network to file `{}`", filename);
   Ok(())
}

const I32_SIZE: usize = std::mem::size_of::<i32>();
const NUM_SIZE: usize = std::mem::size_of::<NumT>();

// rust doesn't have a way to do this generically, so i have to
// copy and paste this function for the 2 different types i need it for
fn consume_i32(list: &[u8]) -> Result<i32, std::io::Error>
{
   Ok(i32::from_be_bytes(list[..I32_SIZE].try_into().map_err(
      |_| make_err("corrupt network depth/input size"),
   )?))
}

fn consume_num(list: &[u8]) -> Result<NumT, std::io::Error>
{
   Ok(NumT::from_be_bytes(
      list[..NUM_SIZE]
         .try_into()
         .map_err(|_| make_err("corrupt weight/bias"))?,
   ))
}

pub fn load_net_from_file(net: &mut NeuralNetwork, filename: &str) -> Result<(), std::io::Error>
{
   let mut bytes = Vec::new();
   {
      let mut file = File::open(filename)?;
      file.read_to_end(&mut bytes)?;
   }

   if !bytes.starts_with(MAGIC_HEADER)
   {
      Err(make_err(
         format!("magic header missing from {}", filename).as_str(),
      ))?;
   }

   let bytes = &bytes[MAGIC_HEADER.len()..];

   let model_layers = consume_i32(&bytes)?;
   let config_layers = net.layers.len() as i32;
   if model_layers != config_layers
   {
      Err(make_err(
         format!(
            "model file is {} layers, config is {} layers",
            model_layers, config_layers
         )
         .as_str(),
      ))?;
   }

   let bytes = &bytes[I32_SIZE..];
   let model_input = consume_i32(&bytes)?;
   let config_input = net
      .layers
      .first()
      .ok_or(make_err("layers empty"))?
      .num_inputs;

   if model_input != config_input
   {
      Err(make_err(
         format!(
            "model file is {} layers, config is {} layers",
            model_input, config_input
         )
         .as_str(),
      ))?;
   }

   let bytes = &bytes[I32_SIZE..];

   for (layer, it) in net.layers.iter().zip(0..)
   {
      let model_out = i32::from_be_bytes(
         bytes[it * I32_SIZE..it * I32_SIZE + I32_SIZE]
            .try_into()
            .map_err(|_| make_err("corrupt layer size"))?,
      );
      if model_out != layer.num_outputs
      {
         Err(make_err(
            format!(
               "model file layer {} is {} outputs, config is {} outputs",
               it, model_out, layer.num_outputs
            )
            .as_str(),
         ))?;
      }
   }

   let bytes = &bytes[I32_SIZE * net.layers.len()..];

   let mut ptr = 0;
   for layer in net.layers.iter_mut()
   {
      for weight in layer.weights.iter_mut()
      {
         *weight = consume_num(&bytes[NUM_SIZE * ptr..])?;
         ptr += 1;
      }

      for bias in layer.biases.iter_mut()
      {
         *bias = consume_num(&bytes[NUM_SIZE * ptr..])?;
         ptr += 1;
      }
   }

   println!("loaded neural network from file `{}`", filename);
   Ok(())
}
