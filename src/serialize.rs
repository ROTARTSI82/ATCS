use crate::config::{make_err, NumT};
use crate::network::NeuralNetwork;
use std::fs::File;
use std::io::{Read, Write};

const MAGIC_HEADER: &'static [u8] = b"ATNeuralNetwork";

pub fn write_net_to_file(net: &NeuralNetwork, filename: &str) -> Result<(), std::io::Error>
{
   let mut bytes = MAGIC_HEADER.to_vec();
   bytes.extend(net.layers.len().to_be_bytes());

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
   Ok(())
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

   let mismatch = "network topology in model file and in config do not match";
   let bytes = &bytes[MAGIC_HEADER.len()..];

   const I32_SIZE: usize = std::mem::size_of::<i32>();
   const NUM_SIZE: usize = std::mem::size_of::<NumT>();

   let consume_i32 = |list: &[u8]| -> Result<i32, std::io::Error> {
      Ok(i32::from_be_bytes(
         bytes[..I32_SIZE]
            .try_into()
            .map_err(|_| make_err("corrupt network depth"))?,
      ))
   };

   let consume_num = |list: &[u8]| -> Result<NumT, std::io::Error> {
      Ok(NumT::from_be_bytes(
         bytes[..NUM_SIZE]
            .try_into()
            .map_err(|_| make_err("corrupt weight/bias"))?,
      ))
   };

   if consume_i32(&bytes)? != net.layers.len() as i32
   {
      Err(make_err(mismatch))?;
   }

   let bytes = &bytes[I32_SIZE..];
   let input = consume_i32(&bytes)?;

   let expected_input = net
      .layers
      .first()
      .ok_or(make_err("layers empty"))?
      .num_inputs;

   if input != expected_input
   {
      Err(make_err(mismatch))?;
   }

   let bytes = &bytes[I32_SIZE..];
   let valid = net.layers.iter().zip(0..).all(|(layer, it)| {
      let raw = bytes[it * I32_SIZE..it * I32_SIZE + I32_SIZE].try_into();
      raw.is_ok() && i32::from_be_bytes(raw.unwrap()) == layer.num_outputs
   });

   if !valid
   {
      Err(make_err(mismatch))?;
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

   Ok(())
}
