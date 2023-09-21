/**
 * main.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Command-line utility to run a neural network according to a
 * configuration file specified as a command-line argument.
 * Optionally, this utility can run training with backpropagation
 * and gradient descent.
 */
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

/**
 * Runs training on a neural network with the specified `train_data`.
 * Configuration parameters for maximum number of iterations, error thresholds, etc. are
 * loaded from the configuration file and stored in the NeuralNetwork struct itself.
 *
 * Keepalive messages will be printed at fixed intervals with loss and learning rate.
 * On exit, summary information about the training session is printed to the console.
 */
fn train_network(network: &mut NeuralNetwork, train_data: &Vec<TrainCase>)
{
   let mut loss = network.err_threshold;
   let mut iteration = 0;

   while iteration < network.max_iters && loss >= network.err_threshold
   {
      loss = 0 as NumT;

      for case in train_data
      {
         network.get_inputs().copy_from_slice(&*case.inp);
         network.feed_forward();
         loss += network.feed_backwards(&*case.outp);
         network.apply_delta_weights();
      }

      if iteration % network.printout_period == 0
      {
         println!(
            "loss={:.6}\tλ={:.6}\tit={}",
            loss, network.learn_rate, iteration
         );
      }

      iteration += 1;
   } // for iteration in 0..network.max_iters

   println!(
      "\nTerminated training after {}/{} iterations",
      iteration, network.max_iters
   );
   println!("loss={:.6}, threshold={:.6}", loss, network.err_threshold);
   if loss < network.err_threshold
   {
      println!("\t+ Met error threshold");
   }

   if iteration == network.max_iters
   {
      println!("\t+ Reached maximum number of iterations");
   }
} // fn train_network(network: &mut NeuralNetwork, train_data: &Vec<TrainCase>)

/**
 * Prints the truth table of the neural network based on the training data.
 * For each training case, it outputs the case, the network's output, and the expected output.
 */
fn print_truth_table(network: &mut NeuralNetwork, train_data: &Vec<TrainCase>)
{
   println!("\nTruth table");
   let mut loss = 0 as NumT;
   for case in train_data
   {
      network.get_inputs().copy_from_slice(&*case.inp);
      network.feed_forward();
      loss += network.feed_backwards(&*case.outp);

      println!(
         "network {:?} = {:?} (expected {:?})",
         case.inp,
         network.get_outputs(),
         case.outp
      );
   } // for case in train_data

   println!("final loss: {}\n", loss);
} // fn print_truth_table(network: &mut Network, train_data: &Vec<TrainCase>)

/**
 * Runs/trains a network according to the configuration file
 * specified as the first command-line argument.
 * If no file is specified, "config.txt" is used by default.
 * On success, Ok(()) is returned. On any error an Err() with
 * a message is returned.
 *
 * Refer to the documentation of `set_and_echo_config` for a specification
 * of the configuration format.
 */
fn main() -> Result<(), Box<dyn Error>>
{
   let args: Vec<_> = std::env::args().collect();
   let filename = args.get(1).map(|x| x.as_str()).unwrap_or("config.txt");
   let config = parse_config(filename)?;

   let mut network = NeuralNetwork::new();
   let mut train_data = Vec::new();
   set_and_echo_config(&mut network, &config, &mut train_data)?;

   if network.do_training
   {
      train_network(&mut network, &train_data);
   }

   print_truth_table(&mut network, &train_data);

   expect_config!(
      Some(Text(filename)),
      config.get("save_file"),
      write_net_to_file(&network, filename.as_str())?
   );

   Ok(())
} // fn main() -> Result<(), io::Error>
