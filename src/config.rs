use crate::config::ConfigValue::{Boolean, FloatList, IntList, Integer, Numeric, Text};
use crate::network::{NetworkLayer, NeuralNetwork, TrainCase};
use crate::serialize::read_net_from_file;
use rand::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::ops::Range;

/// The type of the values contained in the network
/// has been extracted to a type variable for easy modification.
pub type NumT = f64;
pub type FuncT = fn(NumT) -> NumT;

#[derive(Debug)]
pub enum ConfigValue
{
   Text(String),
   Numeric(NumT),
   Integer(i32),
   IntList(Vec<i32>),
   FloatList(Vec<NumT>),
   Boolean(bool),
}

const PARAMS_TO_PRINT: [&str; 6] = [
   "rand_lo",
   "rand_hi",
   "network_topology",
   "max_iterations",
   "learn_rate",
   "error_cutoff",
];

#[macro_export]
macro_rules! expect_config {
   ($name: pat, $str_name: expr, $body: expr) => {
      if let $name = $str_name
      {
         $body
      }
      else
      {
         Err(make_err(concat!("no valid ", stringify!($str_name))))?;
      }
   };
}

pub fn make_err(msg: &str) -> std::io::Error
{
   std::io::Error::new(std::io::ErrorKind::Other, msg)
}

/**
 * Initializes the neural network based off of a configuration map
 * read from the configuration file. If the configuration is invalid,
 * this function will return an Err(std::io::Error) with a message explaining
 * the problem. Otherwise, it will return Ok(()) and set up the network.
 */
pub fn set_and_echo_config(
   net: &mut NeuralNetwork,
   config: &BTreeMap<String, ConfigValue>,
   train_data: &mut Vec<TrainCase>,
) -> Result<(), std::io::Error>
{
   expect_config!(Some(IntList(list)), config.get("network_topology"), {
      net.layers = (0..list.len() - 1)
         .map(|it| NetworkLayer::new(list[it], list[it + 1]))
         .collect();
      net.activations = list
         .iter()
         .map(|it| vec![0 as NumT; *it as usize].into_boxed_slice())
         .collect();

      net.max_width = *list.iter().max().ok_or(make_err("net topology empty"))?;

      let mk_vec = || vec![0 as NumT; net.max_width as usize].into_boxed_slice();
      net.derivs = [mk_vec(), mk_vec()];
   });

   expect_config!(Some(Text(func)), config.get("activation_function"), {
      (net.threshold_func, net.threshold_func_deriv) = match func.as_str()
      {
         "identity" => (ident as FuncT, ident_deriv as FuncT),
         "sigmoid" => (sigmoid as FuncT, sigmoid_deriv as FuncT),
         "tanh" => (tanh as FuncT, tanh_deriv as FuncT),
         _ => Err(make_err(
            "invalid value for key 'activation_function' in config",
         ))?,
      };
   });

   expect_config!(Some(Text(init_mode)), config.get("initialization_mode"), {
      set_initialization_mode(net, init_mode, config)?;
   });

   expect_config!(
      Some(Numeric(lambda)),
      config.get("learn_rate"),
      net.learn_rate = *lambda
   );

   expect_config!(
      Some(Integer(max_iters)),
      config.get("max_iterations"),
      net.max_iters = *max_iters
   );

   expect_config!(
      Some(Integer(printout)),
      config.get("printout_period"),
      net.printout_period = *printout
   );

   expect_config!(
      Some(Numeric(cutoff)),
      config.get("error_cutoff"),
      net.err_threshold = *cutoff
   );

   expect_config!(
      Some(Boolean(train)),
      config.get("do_training"),
      net.do_training = *train
   );

   for (k, v) in config.iter()
   {
      if PARAMS_TO_PRINT.contains(&k.as_str())
      {
         println!("\t{}: {:?}", k, v);
      }

      if net.do_training && k.starts_with("case")
      {
         if let FloatList(outp) = v
         {
            let err_msg = || make_err("invalid test case statement");
            let begin = k.find('[').ok_or(err_msg())? + 1;
            let end = k.rfind(']').ok_or(err_msg())?;
            let sub = &k[begin..end];

            let vec = sub
               .split(',')
               .map(|x| x.trim().parse::<NumT>())
               .collect::<Result<Vec<_>, _>>()
               .map_err(|_| err_msg())?;

            if vec.len() != net.layers.first().unwrap().num_inputs as usize
               || outp.len() != net.layers.last().unwrap().num_outputs as usize
            {
               Err(make_err(
                  "case size does not match configured input/output size",
               ))?;
            }
            else
            {
               train_data.push(TrainCase {
                  inp: vec.into_boxed_slice(),
                  outp: outp.clone().into_boxed_slice(),
               });
            }
         }
         else
         {
            Err(make_err("invalid value for case"))?;
         }
      }
   }
   println!();

   Ok(())
}

/**
 * Loads the weights into a NeuralNetwork based on the method specified in the
 * configuration file.
 */
fn set_initialization_mode(
   net: &mut NeuralNetwork,
   init_mode: &str,
   config: &BTreeMap<String, ConfigValue>,
) -> Result<(), std::io::Error>
{
   match init_mode
   {
      "randomize" =>
      {
         expect_config!(
            (Some(Numeric(hi)), Some(Numeric(lo))),
            (config.get("rand_hi"), config.get("rand_lo")),
            randomize_network(net, *lo..*hi)
         );
      }
      "fixed_value" => todo!("not implemented"),
      "from_file" =>
      {
         expect_config!(
            Some(Text(filename)),
            config.get("load_file"),
            read_net_from_file(net, filename.as_str())?
         );
      }
      _ => Err(make_err("invalid 'initialization_mode' in config"))?,
   }

   Ok(())
}

/**
 * Randomizes the weights of all layers in the neural network to uniform random
 * values in the specified range.
 */
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
} // fn randomize_network(net: &mut NeuralNetwork, range: Range<NumT>)

/**
 * Reads/parses the configuration from the specified file name, returning it as
 * Ok(BTreeMap), which maps from String to ConfigValues.
 *
 * I chose to use a Map instead of a fixed struct so that the configuration format
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

   for (line, line_no) in contents.split('\n').zip(1..)
   {
      let line = line.trim();
      if line.starts_with('#') || line.is_empty()
      {
         continue;
      }

      let err_str = format!("malformed config line {} ({})", line_no, line);
      let err_msg = || make_err(err_str.as_str());

      let key_value_pair: Vec<_> = line.split(':').collect();
      let key = key_value_pair.first().ok_or(err_msg())?.trim().to_string();
      let val = key_value_pair.get(1).ok_or(err_msg())?.trim().to_string();

      if val.starts_with("int[")
      {
         let end = val.rfind(']').ok_or(err_msg())?;
         let list = val["int[".len()..end]
            .split(',')
            .map(|x| x.trim().parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| err_msg())?;

         map.insert(key, IntList(list));
      }
      else if val.starts_with("float[")
      {
         let end = val.rfind(']').ok_or(err_msg())?;
         let list = val["float[".len()..end]
            .split(',')
            .map(|x| x.trim().parse::<NumT>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| err_msg())?;

         map.insert(key, FloatList(list));
      }
      else if val == "true" || val == "false"
      {
         map.insert(key, Boolean(val == "true"));
      }
      else if let Some(stripped) = val.strip_prefix("int")
      {
         let processed_val = stripped.trim().parse::<i32>().map_err(|_| err_msg())?;
         map.insert(key, Integer(processed_val));
      }
      else if let Some(stripped) = val.strip_prefix("float")
      {
         let processed_val = stripped.trim().parse::<NumT>().map_err(|_| err_msg())?;
         map.insert(key, Numeric(processed_val));
      }
      else if val.starts_with('\"')
      {
         let end = val.rfind('\"').ok_or(err_msg())?;
         map.insert(key, Text(String::from(&val[1..end])));
      }
      else
      {
         Err(err_msg())?;
      }
   } // for (line, line_no) in contents.split('\n').zip(1..)

   println!("parsed configuration from file `{}`", filename);
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
