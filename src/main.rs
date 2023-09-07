mod config;
mod network;
mod serialize;

use crate::network::NeuralNetwork;

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
   set_and_echo_config(&mut network, &config)?;

   println!("Network: {:#?}\n", network);

   let test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)];
   let feed_case = |network: &mut NeuralNetwork, case: (i32, i32)| {
      network.get_inputs()[0] = case.0 as NumT;
      network.get_inputs()[1] = case.1 as NumT;

      network.feed_forward();
   };

   if network.do_training
   {
      let mut loss = 0 as NumT;
      for iteration in 0..network.max_iters
      {
         loss = 0 as NumT;
         for (inp, expect_out) in test_cases
         {
            feed_case(&mut network, inp);
            loss += network.feed_backwards(&[expect_out as NumT]);
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
   for (inp, _) in test_cases
   {
      feed_case(&mut network, inp);
      println!("network {:?} = {:?}", inp, network.get_outputs());
   }

   expect_config!(
      Some(Text(filename)),
      config.get("save_file"),
      write_net_to_file(&network, filename.as_str())?
   );

   println!("network: {:#?}", network);

   Ok(())
} // fn main() -> Result<(), io::Error>
