mod config;
mod convolve;
mod network;
mod serialize;

use crate::network::{NeuralNetwork, TrainCase};

extern crate core;

use crate::config::ConfigValue::Text;
use crate::config::{make_err, parse_config, set_and_echo_config, NumT};
use crate::serialize::write_net_to_file;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>
{
   let args: Vec<_> = std::env::args().collect();
   let filename = args.get(1).map(|x| x.as_str()).unwrap_or("config.txt");
   let config = parse_config(filename)?;

   let mut network = NeuralNetwork::new();
   let mut train_data = Vec::new();
   set_and_echo_config(&mut network, &config, &mut train_data)?;

   println!("Network: {:#?}\n", network);

   if network.do_training
   {
      let mut loss = 0 as NumT;
      for iteration in 0..network.max_iters
      {
         loss = 0 as NumT;
         for case in &train_data
         {
            network.get_inputs().copy_from_slice(&*case.inp);
            network.feed_forward();
            loss += network.feed_backwards(&*case.outp);
         }

         if iteration % network.printout_period == 0
         {
            println!("loss: {}", loss);
         }

         if loss < network.err_threshold
         {
            break;
         }
      }

      println!("loss: {}", loss);
   }

   println!("Truth table");
   let mut loss = 0 as NumT;
   for case in &train_data
   {
      network.get_inputs().copy_from_slice(&*case.inp);
      network.feed_forward();
      loss += network.feed_backwards(&*case.outp);

      println!("network {:?} = {:?} (expected {:?})", case.inp, network.get_outputs(), case.outp);
   }

   println!("final loss: {}", loss);

   expect_config!(
      Some(Text(filename)),
      config.get("save_file"),
      write_net_to_file(&network, filename.as_str())?
   );

   // println!("network: {:#?}", network);

   Ok(())
} // fn main() -> Result<(), io::Error>
