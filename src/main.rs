mod config;
mod network;
mod serialize;

use crate::network::NeuralNetwork;

extern crate core;

use crate::config::ConfigValue::{Boolean, IntList, Integer, Numeric, Text};
use crate::config::{parse_config, set_and_echo_config, NumT};
use crate::serialize::{write_net_to_file, load_net_from_file};
use std::error::Error;
use std::io::Read;
use std::io::Write;
use std::iter::{Flatten, Map};
use std::slice::Iter;

fn main() -> Result<(), Box<dyn Error>>
{
   let args: Vec<_> = std::env::args().collect();
   let filename = args.get(1).map(|x| x.as_str()).unwrap_or("config.txt");
   let mut config = parse_config(filename)?;

   let mut network = NeuralNetwork::new();
   set_and_echo_config(&mut network, &mut config)?;

   for (k, v) in config.iter()
   {
      if k.starts_with("set")
      {
         let indices: Vec<_> = k["set".len()..]
            .split(",")
            .map(|s| s.trim().parse::<usize>().unwrap())
            .collect();
         if let Numeric(x) = v
         {
            if indices[1] == 0
            {
               *network.layers[indices[0]].get_weight_mut(indices[2] as i32, indices[3] as i32) =
                  *x;
            }
            else
            {
               network.layers[indices[0]].biases[indices[2]] = *x;
            }
         }
         else
         {
            panic!("invalid set directive");
         }
      }
   }

   println!("Network: {:?}", network);
   println!("Truth table: ");

   for it in [(0, 0), (0, 1), (1, 0), (1, 1)]
   {
      network.feed_forward(&[it.0 as NumT, it.1 as NumT]);
      println!("network {:?} = {:?}", it, network.get_outputs());
   }

   if let Some(Text(filename)) = config.get("save_file") {
      write_net_to_file(&network, filename.as_str())?;
   }

   Ok(())
} // fn main() -> Result<(), io::Error>
