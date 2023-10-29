/**
 * main.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Command-line utility to run a neural network according to a
 * configuration file specified as a command-line argument.
 * Optionally, this utility can run training with backpropagation
 * and backpropagation.
 *
 * See documentation for `parse_config`, `set_and_echo_config`,
 * and `load_dataset_from_config_txt` in `config.rs`
 * for more info about the configuration format.
 */
mod config;
mod network;
mod serialize;

extern crate core;

use crate::config::ConfigValue::Text;
use crate::config::*;
use crate::network::{Datapoint, NeuralNetwork};
use crate::serialize::write_net_to_file;
use std::error::Error;

/**
 * Runs training on a neural network with the specified `dataset`.
 * Configuration parameters for maximum number of iterations, error thresholds, etc. are
 * loaded from the configuration file and stored in the NeuralNetwork struct itself.
 *
 * Keepalive messages will be printed at fixed intervals with loss and learning rate.
 * On exit, summary information about the training session is printed to the console.
 */
fn train_network(network: &mut NeuralNetwork, dataset: &Vec<Datapoint>)
{
   let mut loss = network.error_cutoff;
   let mut iteration = 0;

   let start = std::time::Instant::now();
   while iteration < network.max_iterations && loss >= network.error_cutoff
   {
      loss = 0.0;

      for case in dataset
      {
         network.get_inputs().copy_from_slice(&case.inputs);
         network.feed_forward::<true>();
         loss += network.feed_backward(&case.expected_outputs);
      }

      loss /= dataset.len() as NumT;
      if iteration % network.printout_period == 0
      {
         println!("loss={:.6}\tλ={:.6}\tit={}",
                  loss, network.learn_rate, iteration);
      }

      iteration += 1;
   } // while iteration < network.max_iterations && loss >= network.error_cutoff
   let duration = start.elapsed();

   println!("\nTerminated training after {}/{} iterations",
            iteration, network.max_iterations);

   println!("loss={:.6}, threshold={:.6}", loss, network.error_cutoff);

   if loss < network.error_cutoff
   {
      println!("\t+ Met error threshold");
   }

   if iteration == network.max_iterations
   {
      println!("\t+ Reached maximum number of iterations");
   }

   let ms = (duration.as_micros() as NumT) / 1000.0;
   println!("\nTrained in {:.4} seconds ({:.4}ms per epoch, {:.4}ms per case)",
            duration.as_secs_f64(),
            ms / iteration as NumT,
            ms / (dataset.len() as NumT * iteration as NumT));
} // fn train_network(network: &mut NeuralNetwork, dataset: &Vec<Datapoint>)

/**
 * Prints the truth table of the neural network based on the training data.
 * For each training case, it outputs the case, the network's output, and the expected output.
 */
fn print_truth_table(network: &mut NeuralNetwork, dataset: &Vec<Datapoint>)
{
   println!("\nTruth table");
   let mut loss = 0.0;

   let start = std::time::Instant::now();
   for case in dataset
   {
      network.get_inputs().copy_from_slice(&case.inputs);
      network.feed_forward::<false>();
      loss += network.calculate_error(&case.expected_outputs);

      println!("network {:.3?} = {:.3?} (expected {:.3?})",
               case.inputs,
               network.get_outputs(),
               case.expected_outputs);
   } // for case in dataset

   let diff = start.elapsed();
   let ms = (diff.as_micros() as NumT) / 1000.0;
   println!("Ran epoch in {:.4}ms ({:.4}ms per case)", ms, ms / dataset.len() as NumT);

   loss /= dataset.len() as NumT;
   println!("final loss: {}\n", loss);
} // fn print_truth_table(network: &mut Network, dataset: &Vec<Datapoint>)

/**
 * Runs/trains a network according to the configuration file
 * specified as the first command-line argument.
 * If no file is specified, "config.txt" is used by default.
 * On success, Ok(()) is returned. On any error, an Err() with
 * an appropriate message is returned.
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
   set_and_echo_config(&mut network, &config)?;

   // this dataset is actually used both as the test set and the training set.
   let mut dataset = Vec::new();
   load_dataset_from_config_txt(&network, &config, &mut dataset)?;

   if network.do_training
   {
      train_network(&mut network, &dataset);
   }

   print_truth_table(&mut network, &dataset);

   expect_config!(Some(Text(filename)),
                  config.get("save_file"),
                  write_net_to_file(&network, filename.as_str())?);

   Ok(())
} // fn main() -> Result<(), io::Error>
