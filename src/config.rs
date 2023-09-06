use crate::config::ConfigValue::{Boolean, IntList, Integer, Numeric, Text};
use crate::network::{NetworkLayer, NeuralNetwork};
use rand::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::ops::Range;

/// The type of the values contained in the network
/// has been extracted to a type variable for easy modification.
pub type NumT = f64;
pub type FuncT = fn(NumT) -> NumT;

pub fn make_err(msg: &str) -> std::io::Error
{
   std::io::Error::new(std::io::ErrorKind::Other, msg)
}

#[derive(Debug)]
pub enum ConfigValue
{
   Text(String),
   Numeric(NumT),
   Integer(i32),
   IntList(Vec<i32>),
   Boolean(bool),
}

/**
 * Initializes the neural network based off of a configuration map
 * read from the configuration file. If the configuration is invalid,
 * this function will return an Err(std::io::Error) with a message explaining
 * the problem. Otherwise, it will return Ok(()) and set up the network.
 */
pub fn set_and_echo_config(
   net: &mut NeuralNetwork,
   config: &mut BTreeMap<String, ConfigValue>,
) -> Result<(), std::io::Error>
{
   println!("Loaded configuration");
   if let Some(IntList(list)) = config.get("network_topology")
   {
      net.layers = (0..list.len() - 1)
         .map(|it| NetworkLayer::new(list[it], list[it + 1]))
         .collect();
      net.activations = list
         .iter()
         .skip(1)
         .map(|it| vec![0 as NumT; *it as usize].into_boxed_slice())
         .collect();

      println!("\tnetwork_topology: {:?}", list);
      config.remove("network_topology");
   }
   else
   {
      Err(make_err("config requires IntList 'network_topology'"))?;
   }

   if let Some(Text(func)) = config.get("activation_function")
   {
      (net.threshold_func, net.threshold_func_deriv) = match func.as_str()
      {
         "identity" => (ident as FuncT, ident_deriv as FuncT),
         "sigmoid" => (sigmoid as FuncT, sigmoid_deriv as FuncT),
         "tanh" => (tanh as FuncT, tanh_deriv as FuncT),
         _ => Err(make_err(
            "invalid value for key 'activation_function' in config",
         ))?,
      };

      println!("\tactivation_function: {:?}", func);
      config.remove("activation_function");
   }
   else
   {
      Err(make_err("no valid 'activation_function' in config"))?;
   }

   if let Some(Text(init_mode)) = config.get("initialization_mode")
   {
      match init_mode.as_ref()
      {
         "randomize" =>
         {
            if let (Some(Numeric(hi)), Some(Numeric(lo))) =
               (config.get("rand_hi"), config.get("rand_lo"))
            {
               randomize_network(net, *lo..*hi);
            }
            else
            {
               Err(make_err("invalid 'rand_hi' and 'rand_lo' range"))?;
            }
         }
         "fixed_value" => todo!("not implemented"),
         "from_file" =>
         {
            if let Some(Text(filename)) = config.get("load_file")
            {
               todo!("not implemented")
            }
            else
            {
               Err(make_err("no valid 'load_file' filename in config"))?;
            }
         }
         _ => Err(make_err("invalid 'initialization_mode' in config"))?,
      }
   }
   else
   {
      Err(make_err("no valid 'initialization_mode' in config"))?;
   }

   for (k, v) in config.iter()
   {
      println!("unknown value {}: {:?} in config", k, v);
      // Err(make_err(
      //    format!("unrecognized value {}: {:?} in config", k, v).as_ref(),
      // ))?;
   }

   Ok(())
}

fn randomize_network(net: &mut NeuralNetwork, range: Range<NumT>)
{
   let mut rng = rand::thread_rng();
   for layer in net.layers.iter_mut()
   {
      for bias in layer.biases.iter_mut()
      {
         *bias = rng.gen_range(range.clone());
      }

      for weight in layer.weights.iter_mut()
      {
         *weight = rng.gen_range(range.clone());
      }
   }
}

/**
 * Reads/parses the configuration from the specified file name, returning it as
 * Ok(BTreeMap), which maps from String to ConfigValues. I chose
 * to use a Map instead of a fixed struct so that the configuration format
 * is more easily extensible, and I reduce the number of if statements needed
 * to write each value into the correct place. All details about the configuration's
 * structure could thus be extracted from the parsing step into a
 * a separate function.
 *
 * If there is an error with opening/reading the file or a problem
 * within the configuration file itself, this function prematurely returns
 * an Err(io::Error) with a message specifying the problem.
 */
pub fn parse_config(filename: &str) -> Result<BTreeMap<String, ConfigValue>, std::io::Error>
{
   let mut file = File::open(filename)?;
   let mut map = BTreeMap::new();

   let mut contents = String::new();
   file.read_to_string(&mut contents)?;

   for (line, line_no) in contents.split("\n").zip(1..)
   {
      let line = line.trim();
      if line.starts_with('#') || line.is_empty()
      {
         continue;
      }

      let err_str = format!("malformed config line {} ({})", line_no, line);
      let err_msg = || make_err(err_str.as_str());

      let key_value_pair: Vec<_> = line.split(":").collect();
      let key = key_value_pair.get(0).ok_or(err_msg())?.trim().to_string();
      let val = key_value_pair.get(1).ok_or(err_msg())?.trim().to_string();

      if val.starts_with("[")
      {
         let end = val.rfind("]").ok_or(err_msg())?;
         let list = val[1..end]
            .split(",")
            .map(|x| x.trim().parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| err_msg())?;

         map.insert(key, IntList(list));
      }
      else if val == "true" || val == "false"
      {
         map.insert(key, Boolean(val == "true"));
      }
      else if val.starts_with("int")
      {
         let processed_val = val["int".len()..]
            .trim()
            .parse::<i32>()
            .map_err(|_| err_msg())?;
         map.insert(key, Integer(processed_val));
      }
      else if val.starts_with("float")
      {
         let processed_val = val["float".len()..]
            .trim()
            .parse::<NumT>()
            .map_err(|_| err_msg())?;
         map.insert(key, Numeric(processed_val));
      }
      else if val.starts_with("\"")
      {
         let end = val.rfind("\"").ok_or(err_msg())?;
         map.insert(key, Text(String::from(&val[1..end])));
      }
      else
      {
         Err(err_msg())?;
      }
   } // for (line, line_no) in contents.split('\n').zip(1..)

   Ok(map)
} // fn load_config(&str) -> Result<HashMap<String, ConfigValue>, io::Error>

pub fn ident(x: NumT) -> NumT
{
   x
}

pub fn ident_deriv(_: NumT) -> NumT
{
   1 as NumT
}

fn sigmoid(x: NumT) -> NumT
{
   1 as NumT / (1 as NumT + (-x).exp())
}

fn sigmoid_deriv(x: NumT) -> NumT
{
   x * (1 as NumT - x)
}

fn tanh(x: NumT) -> NumT
{
   x.tanh()
}

fn tanh_deriv(x: NumT) -> NumT
{
   // tanh'(x) = sech(x)^2 = 1 - tanh(x)^2
   1 as NumT - x * x
}
