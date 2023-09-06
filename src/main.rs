mod config;
mod network;
mod serialize;

use crate::network::NeuralNetwork;

extern crate core;

use crate::config::ConfigValue::{Text};
use crate::config::{parse_config, set_and_echo_config, NumT};
use crate::serialize::{write_net_to_file};
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

   for it in [(0, 0), (0, 1), (1, 0), (1, 1)]
   {
      network.feed_forward(&[it.0 as NumT, it.1 as NumT]);
      println!("network {:?} = {:?}", it, network.get_outputs());
   }

   if let Some(Text(filename)) = config.get("save_file")
   {
      write_net_to_file(&network, filename.as_str())?;
   }

   Ok(())
} // fn main() -> Result<(), io::Error>
