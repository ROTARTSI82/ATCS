/**
 * config.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * This file exports utility functions used for parsing the configuration file
 * and initializing the neural network based on the specified configurations.
 *
 * The `ConfigValue` union type is defined to store the values
 * from the configuration file (whether it is text, an integer, a float, or a list).
 *
 * Types:
 * `NumT` (for the type of network weights and values),
 * `FuncT` (for activation functions and their derivatives)
 *
 * Table of contents:
 * ```pub fn set_and_echo_config(net: &mut NeuralNetwork, config: &BTreeMap<String, ConfigValue>)
 *                               -> Result<(), std::io::Error>```
 * ```pub fn load_dataset_from_config_txt(net: &NeuralNetwork,
 *                                        config: &BTreeMap<String, ConfigValue>,
 *                                        dataset_out: &mut Vec<Datapoint>)
 *                                        -> Result<(), std::io::Error>```
 * ```fn set_initialization_mode(net: &mut NeuralNetwork, init_mode: &str,
 *                               config: &BTreeMap<String, ConfigValue>)
 *                               -> Result<(), std::io::Error>```
 * `fn randomize_network(net: &mut NeuralNetwork, range: Range<NumT>)`
 * `pub fn parse_config(filename: &str) -> Result<BTreeMap<String, ConfigValue>, std::io::Error>`
 * `pub fn sigmoid(x: NumT) -> NumT`
 * `pub fn sigmoid_deriv(x: NumT) -> NumT`
 * `pub fn ident(x: NumT) -> NumT`
 * `pub fn ident_deriv(x: NumT) -> NumT`
 * `pub fn leaky_relu(x: NumT) -> NumT`
 * `pub fn leaky_relu_deriv(x: NumT) -> NumT`
 * `pub fn tanh(x: NumT) -> NumT`
 * `pub fn tanh_deriv(x: NumT) -> NumT`
 */
use crate::config::ConfigValue::{Boolean, FloatList, IntList, Integer, Numeric, Text};
use crate::network::{Datapoint, NetworkLayer, NeuralNetwork};
use crate::serialize::{load_dataset_from_file, read_net_from_file};

use rand::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use rand_distr::StandardNormal;

/**
 * The type of the values contained in the network
 * has been extracted to a type variable for easy modification.
 */
pub type NumT = f32;
pub type FuncT = fn(NumT) -> NumT;

/**
 * A union of the possible values contained in the configuration file.
 * This type is used to store the configuration in a key:value map.
 */
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

/**
 * These are the parameters of the configuration that are printed out
 * before running the network.
 */
const PARAMS_TO_PRINT: [&str; 7] = ["rand_lo",
   "rand_hi",
   "network_topology",
   "max_iterations",
   "learn_rate",
   "error_cutoff",
   "activation_function"];

/**
 * This macro attempts to unwrap a certain value (specified by the expression `str_name`)
 * from the configuration according to the pattern specified by `name`.
 * An error with a helpful message if unwrapping fails, and if it is successful,
 * the code in `body` is executed.
 *
 * For examples, see usages in `set_and_echo_config`
 */
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
} // macro_rules! expect_config

/// Utility method for quickly constructing an Error object from a string message.
#[inline]
pub fn make_err(msg: &str) -> std::io::Error
{
   std::io::Error::new(std::io::ErrorKind::Other, msg)
}

/**
 * Initializes the neural network based off of a configuration map and
 * prints out the important parameters onto the console (specified in PARAMS_TO_PRINT)
 *
 * If the configuration is invalid, this function will return an Err(std::io::Error)
 * with a message explaining the problem. Otherwise, it will return Ok(()) and set up the network.
 *
 * `network_topology` MUST be a `IntList` specifying the output sizes
 * of each layer. The first element must be the number of inputs to the network,
 * and each subsequent value must be the number of outputs for the corresponding layer,
 * including the final output layer.
 *
 * `do_training` MUST be a `Boolean` specifying whether to train the network.
 *
 * If `do_training` is true, the following values are also required:
 * `Integer` values of `max_iterations` and `printout_period`,
 * and `Numeric` values of `learn_rate` and `error_cutoff`
 *
 * `max_iterations` is used to determine when to stop training.
 * One iteration consists of processing all test cases once (also known as an 'epoch').
 * `printout_period` is the period (in number of iterations) for printing keepalive messages.
 * Every `printout_period` iterations, the loss and current step size are printed.
 * `learn_rate` is the step size for backpropagation, and `error_cutoff` is the
 * minimum loss past which we stop training.
 * Loss is defined as the sum of 0.5 * (difference between output and expected values)^2 across
 * all the output nodes and across all the test cases.
 *
 * `activation_function` MUST be `Text` with value of `"sigmoid"`, `"identity"`, `"tanh"`, or
 * `"leaky_relu"`. It specifies the threshold function to apply element-wise
 * on the output of each layer.
 *
 * `initialization_mode` MUST be `Text` with value of `"from_file"`,
 * `"randomize"`, or `"fixed_value"`. See `randomize_network` for `"randomize"`.
 * The file used in `"from_file"` is specified by the `load_file` `Text` field.
 * The range used by `"randomize"` is specified by the `Numeric` values `rand_lo` and `rand_hi`.
 *
 * `save_file` must be `Text` of the path in which to save the resulting network.
 * The binary format is used.
 *
 * Refer to the documentation of `parse_config` for the syntax of the configuration format.
 */
pub fn set_and_echo_config(net: &mut NeuralNetwork, config: &BTreeMap<String, ConfigValue>)
                           -> Result<(), std::io::Error>
{
   expect_config!(Some(Boolean(train)),
                  config.get("do_training"),
                  net.do_training = *train);

   expect_config!(Some(IntList(list)), config.get("network_topology"), 
   {
      net.layers =
         (0..list.len() - 1).map(|it| NetworkLayer::new(list[it], list[it + 1], net.do_training))
                            .collect();
      net.activations = list.iter()
                            .map(|it| vec![0.0; *it as usize].into_boxed_slice())
                            .collect();

      let max_width = *list.iter().max().ok_or(make_err("net topology empty"))? as usize;
      let mk_vec = || vec![0.0; max_width].into_boxed_slice();
      net.omegas = [mk_vec(), mk_vec()];
   }); // expect_config! Some(IntList(list)), config.get("network_topology")

   expect_config!(Some(Text(func)), config.get("activation_function"), 
   {
      (net.threshold_func, net.threshold_func_deriv) = match func.as_str()
      {
         "identity" => (ident as FuncT, ident_deriv as FuncT),
         "sigmoid" => (sigmoid as FuncT, sigmoid_deriv as FuncT),
         "tanh" => (tanh as FuncT, tanh_deriv as FuncT),
         "leaky_relu" => (leaky_relu as FuncT, leaky_relu_deriv as FuncT),
         _ => Err(make_err("invalid value for key 'activation_function' in config"))?,
      };
   }); // expect_config! Some(Text(func)), config.get("activation_function")

   expect_config!(Some(Text(init_mode)),
                  config.get("initialization_mode"),
                  set_initialization_mode(net, init_mode, config)?);

   if net.do_training
   {
      expect_config!(Some(Numeric(lambda)),
                     config.get("learn_rate"),
                     net.learn_rate = *lambda);

      expect_config!(Some(Numeric(decay)), config.get("learn_decay"), net.learn_decay = *decay);
      expect_config!(Some(Numeric(noise)), config.get("add_noise"), net.add_noise = *noise);
      expect_config!(Some(Numeric(drop)), config.get("input_dropout"), net.dropout = *drop);

      expect_config!(Some(Integer(max_iters)),
                     config.get("max_iterations"),
                     net.max_iterations = *max_iters);

      expect_config!(Some(Integer(printout)),
                     config.get("printout_period"),
                     net.printout_period = *printout);

      expect_config!(Some(Numeric(cutoff)),
                     config.get("error_cutoff"),
                     net.error_cutoff = *cutoff);
   } // if net.do_training

   let filtered = config.iter()
         .filter(|(key, _)| PARAMS_TO_PRINT.contains(&key.as_str()));
   for (key, value) in filtered
   {
      println!("\t{}: {:?}", key, value);
   }

   Ok(())
} // fn set_and_echo_config(net: &mut NeuralNetwork, config: ...) -> Result<...>

/**
 * Loads a dataset stored in the configuration text file itself, in the format
 * of `case [...]` keys. This dataset is used both for training and testing.
 *
 * This function returns Ok(()) and writes the cases into `dataset` on success,
 * otherwise, Err(io::Error) is returned and `dataset` may be partially filled.
 *
 * `case [...]` keys are also loaded into `dataset` as test cases for the network.
 * For example: `case [1.0, 2.4]: float[3.6, 2.6]` specifies the input and the expected output
 * as a `FloatList`. The array in the `case [..]` key are parsed in the same way with
 * rust parsing, so e-notation is supported in all cases.
 */
pub fn load_dataset_from_config_txt(net: &mut NeuralNetwork, config: &BTreeMap<String, ConfigValue>,
                                    dataset_out: &mut Vec<Datapoint>)
                                    -> Result<(), std::io::Error>
{
   let expected = (net.layers.first().unwrap().num_inputs as usize,
                   net.layers.last().unwrap().num_outputs as usize);

   expect_config!(Some(Text(mode)), config.get("dataset_mode"), {
      if mode == "from_file"
      {
         expect_config!(Some(Text(filename)), config.get("dataset_file"), {
            load_dataset_from_file(dataset_out, filename, expected)?;
            return Ok(());
         });
      } else if mode != "truth_table"
      {
         Err(make_err("unrecognized `dataset_mode` config value"))?;
      }
   });

   for (key, value) in config.iter().filter(|(key, _)| key.starts_with("case"))
   {
      if let FloatList(outp) = value
      {
         let err_msg = || make_err("invalid test case statement");
         let begin = key.find('[').ok_or(err_msg())? + 1;
         let end = key.rfind(']').ok_or(err_msg())?;
         let sub = &key[begin..end];

         let vec = sub.split(',')
                      .map(|x| x.trim().parse::<NumT>())
                      .collect::<Result<Vec<_>, _>>()
                      .map_err(|_| err_msg())?;

         if vec.len() != expected.0 || outp.len() != expected.1
         {
            Err(make_err("case size does not match configured input/output size"))?;
         }
         else
         {
            dataset_out.push(Datapoint { inputs: vec.into_boxed_slice(),
                                         expected_outputs: outp.clone().into_boxed_slice() });
         }
      } // if let FloatList(outp) = value
      else
      {
         Err(make_err("invalid value for case"))?;
      }
   } // for (key, value) in config.iter().filter(...)

   println!();
   Ok(())
} // fn load_dataset_from_config_txt(: &NeuralNetwork, config, : &mut Vec<Datapoint>) -> Result<...>

/**
 * Loads the weights into a NeuralNetwork based on the method specified by `init_mode`
 * and the additional data in the `config` map.
 *
 * This function supports loading weights from a binary file, or setting them to
 * uniform random values within a specified range (in `config`).
 * Currently, setting the network weights to a fixed value is not yet implemented.
 *
 * If the `Boolean` `print_weights` config option is set to true,
 * the weights of the network are printed to the console after loading them.
 */
fn set_initialization_mode(net: &mut NeuralNetwork, init_mode: &str,
                           config: &BTreeMap<String, ConfigValue>)
                           -> Result<(), std::io::Error>
{
   match init_mode
   {
      "randomize" =>
      {
         expect_config!((Some(Numeric(hi)), Some(Numeric(lo))),
                        (config.get("rand_hi"), config.get("rand_lo")),
                        randomize_network(net, *lo..*hi));
      }
      "smart_random" =>
      {
         expect_config!(Some(Numeric(gain)), config.get("gain"), net.gain = *gain);
         smart_random(net);
      },
      "fixed_value" => todo!("not implemented"),
      "from_file" =>
      {
         expect_config!(Some(Text(filename)),
                        config.get("load_file"),
                        read_net_from_file(net, filename.as_str())?);
      }
      _ => Err(make_err("invalid 'initialization_mode' in config"))?,
   } // match init_mode

   expect_config!(Some(Boolean(print)), config.get("print_weights"),
   {
      if *print
      {
         for (layer_no, layer) in net.layers.iter().enumerate()
         {
            println!("===== Layer {} weights =====", layer_no);
            for col in layer.weights.chunks(layer.num_outputs as usize)
            {
               println!("{:#.3?}", col);
            }
         }
      }
   }); // expect_config!(Some(Boolean(print)), ...)

   Ok(())
} // fn set_initialization_mode(: &mut NeuralNetwork, init_mode: &str, config: ...) -> Result<...>

/**
 * Randomizes the weights of all layers in the neural network to uniform random
 * values in the specified range.
 */
fn randomize_network(net: &mut NeuralNetwork, range: Range<NumT>)
{
   let mut rng = thread_rng();
   for layer in net.layers.iter_mut()
   {
      for weight in layer.weights.iter_mut()
      {
         *weight = rng.gen_range(range.clone());
      }
   }
} // fn randomize_network(net: &mut NeuralNetwork, range: Range<NumT>)

fn smart_random(net: &mut NeuralNetwork)
{
   let mut rng = thread_rng();
   let gain = net.gain;

   for layer in net.layers.iter_mut()
   {
      for weight in layer.weights.iter_mut()
      {
         *weight = gain * rng.sample::<NumT,_>(StandardNormal) / layer.num_inputs as NumT;
      }
   }
}

/**
 * Reads/parses the configuration from the specified file name, returning it as
 * Ok(BTreeMap), which maps from String to ConfigValues.
 *
 * Lines starting with '#' are ignored as comments, as are blank lines.
 * Every other line should be a key-value pair separated by a colon ("key: value").
 * Keys can be any string that does not contain a colon, and values can be
 * the types supported by `ConfigValue`.
 *
 * `Text` values are written as `"Hello World!"`. No escape sequences are implemented.
 * `IntList` values are written as `int[1,2]`
 * `FloatList` values are written as `float[1,2]`. Rust parsing is used, so e-notation is supported.
 * `Integer` values are written as `int 1`
 * `Numeric` values are written as `float 3.14`. Again, e-notation is supported with rust parsing.
 * `Boolean` values are either `true` or `false`
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

   let lines = contents.split('\n')
                       .zip(1..)
                       .map(|(line, line_no)| (line.trim(), line_no))
                       .filter(|(line, _)| !line.starts_with('#') && !line.is_empty());

   for (line, line_no) in lines
   {
      let err_str = format!("malformed config line {} ({})", line_no, line);
      let err_msg = || make_err(err_str.as_str());

      let key_value_pair: Vec<_> = line.split(':').collect();
      let key = key_value_pair.first().ok_or(err_msg())?.trim().to_string();
      let val = key_value_pair.get(1).ok_or(err_msg())?.trim().to_string();

      if val.starts_with("int[")
      {
         let end = val.rfind(']').ok_or(err_msg())?;
         let list = val["int[".len()..end].split(',')
                                          .map(|x| x.trim().parse::<i32>())
                                          .collect::<Result<Vec<_>, _>>()
                                          .map_err(|_| err_msg())?;

         map.insert(key, IntList(list));
      }
      else if val.starts_with("float[")
      {
         let end = val.rfind(']').ok_or(err_msg())?;
         let list = val["float[".len()..end].split(',')
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

/**
 * These are the supported activation functions along with their derivatives:
 * The identity function, hyperbolic tangent, logistic sigmoid, and leaky ReLU.
 */

fn sigmoid(x: NumT) -> NumT
{
   1.0 / (1.0 + (-x).exp())
}

fn sigmoid_deriv(x: NumT) -> NumT
{
   let f_of_x = sigmoid(x);
   f_of_x * (1.0 - f_of_x)
}

pub fn ident(x: NumT) -> NumT
{
   x
}

pub fn ident_deriv(_: NumT) -> NumT
{
   1.0
}

fn leaky_relu(x: NumT) -> NumT
{
   leaky_relu_deriv(x) * x
}

fn leaky_relu_deriv(x: NumT) -> NumT
{
   if x > 0.0
   {
      1.0
   }
   else
   {
      0.01
   }
}

fn tanh(x: NumT) -> NumT
{
   x.tanh()
}

fn tanh_deriv(x: NumT) -> NumT
{
   let cosh = x.cosh();
   1.0 / (cosh * cosh)
}
