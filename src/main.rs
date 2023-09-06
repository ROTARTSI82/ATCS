mod config;
mod network;
mod serialize;

use crate::network::NeuralNetwork;

extern crate core;

use crate::config::ConfigValue::Text;
use crate::config::{parse_config, set_and_echo_config, NumT};
use crate::serialize::write_net_to_file;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>
{
   let args: Vec<_> = std::env::args().collect();
   let filename = args.get(1).map(|x| x.as_str()).unwrap_or("config.txt");
   let mut config = parse_config(filename)?;

   let mut network = NeuralNetwork::new();
   set_and_echo_config(&mut network, &mut config)?;

   println!("Network: {:#?}\n", network);
   println!("Truth table: ");

   let conv = |b| if b
   {
      1 as NumT
   }
   else
   {
      0 as NumT
   };

   let mut best = 100 as NumT;
   let mut loss = 0f64;
   for it in 0..100000
   {
      loss = 0f64;
      for a in 0..2 {
         for b in 0..2 {
            let a = (a == 1);
            let b = b == 1;
            network.get_inputs()[0] = conv(a);
            network.get_inputs()[1] = conv(b);
            let expect = [conv(!a && b || !b && a)];
            network.feed_forward();
            loss += network.feed_backwards(&expect);
         }
      }

      if loss < best && it % 1000 == 0
      {
         println!("loss: {}", loss);
         for it in [(0, 0), (0, 1), (1, 0), (1, 1)]
         {
            network.get_inputs()[0] = it.0 as NumT;
            network.get_inputs()[1] = it.1 as NumT;
            network.feed_forward();
            println!("network {:?} = {:?};", it, network.get_outputs());
         }
         best = loss;
      }
   }

   if let Some(Text(filename)) = config.get("save_file")
   {
      write_net_to_file(&network, filename.as_str())?;
   }

   println!("network: {:#?} loss = {}", network, loss);

   Ok(())
} // fn main() -> Result<(), io::Error>
