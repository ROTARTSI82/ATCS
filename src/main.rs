/**
 * main.rs
 * By Grant Yang
 * Created on 2023.09.05
 *
 * Command-line utility to run a neural network according to a
 * configuration file specified as a command-line argument.
 * Optionally, this utility can run training with backpropagation.
 *
 * See documentation for `parse_config`, `set_and_echo_config`,
 * and `load_dataset_from_config_txt` in `config.rs`
 * for more info about the configuration format.
 *
 * Table of Contents:
 * `fn train_network(network: &mut NeuralNetwork, dataset: &Vec<Datapoint>)`
 * `fn print_truth_table(network: &mut NeuralNetwork, dataset: &Vec<Datapoint>)`
 * `fn main() -> Result<(), Box<dyn Error>>`
 */
mod config;
mod network;
mod serialize;

extern crate core;

use crate::config::ConfigValue::{Numeric, Text};
use crate::config::*;
use crate::network::{Datapoint, NeuralNetwork};
use crate::serialize::write_net_to_file;
use rand::prelude::IteratorRandom;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/**
 * Get index of maximal element
 */
fn max_idx(arr: &[NumT]) -> Option<usize>
{
   Some(arr.iter()
           .enumerate()
           .max_by(|(_, i), (_, j)| i.total_cmp(j))?
           .0)
}

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
   let mut rng = rand::thread_rng();
   let mut loss = network.error_cutoff;
   let mut iteration = 0;
   let mut loss_log = Vec::new();

   let start = std::time::Instant::now();
   while iteration < network.max_iterations && loss >= network.error_cutoff
   {
      loss = 0.0;

      let mut correct = 0;
      for case in dataset.iter()
                         .choose_multiple(&mut rng, network.batch_size as usize)
      {
         network.randomize_dropouts();
         network.get_inputs().copy_from_slice(&case.inputs);
         network.add_noise(&mut rng);
         network.feed_forward();
         loss += network.feed_backward(&case.expected_outputs, iteration + 1);

         let (a, b) =
            (max_idx(network.get_outputs()).unwrap(), max_idx(&case.expected_outputs).unwrap());
         //println!("{:#?} vs. {:#?} {a} {b}", network.get_outputs(), &case.expected_outputs);
         if a == b
         {
            correct += 1
         }
      }

      network.apply_deltas();

      loss /= network.batch_size as NumT;
      loss_log.push(loss);
      if iteration % network.printout_period == 0
      {
         println!("loss={:.6}\tacc={:.2}\tλ={:.6}\tit={}",
                  loss,
                  100.0 * correct as NumT / network.batch_size as NumT,
                  network.train_params.learn_rate,
                  iteration);
      }

      iteration += 1;
   } // while iteration < network.max_iterations && loss >= network.error_cutoff
   let duration = start.elapsed();

   println!("\nTerminated training after {}/{} iterations",
            iteration, network.max_iterations);

   println!("loss={:.6}, threshold={:.6}", loss, network.error_cutoff);
   let mut file = File::create("loss.csv").unwrap();
   file.write_all(&loss_log.into_iter()
                           .map(|x| (x.to_string() + ",").into_bytes())
                           .flatten()
                           .collect::<Vec<_>>()
                           .into_boxed_slice())
       .unwrap();

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
   network.zero_dropouts();
   println!("\nTruth table");
   let mut loss = 0.0;

   let mut correct = 0;

   let start = std::time::Instant::now();
   for case in dataset
   {
      network.get_inputs().copy_from_slice(&case.inputs);
      network.feed_forward();
      loss += network.calculate_error(&case.expected_outputs);

      println!("network = {:?} (expected {:.2?})",
               network.get_outputs(),
               case.expected_outputs);

      let (a, b) =
         (max_idx(network.get_outputs()).unwrap(), max_idx(&case.expected_outputs).unwrap());
      if a == b
      {
         correct += 1
      }
   } // for case in dataset

   let diff = start.elapsed();
   let ms = (diff.as_micros() as NumT) / 1000.0;
   println!("Ran epoch in {:.4}ms ({:.4}ms per case)",
            ms,
            ms / dataset.len() as NumT);

   loss /= dataset.len() as NumT;
   println!("final loss: {}\tfinal accuracy: {}\n",
            loss,
            100.0 * correct as NumT / dataset.len() as NumT);
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
   load_dataset_from_config_txt(&mut network, &config, &mut dataset)?;

   expect_config!((Some(Numeric(label0)), Some(Numeric(label1))),
                  (config.get("label0"), config.get("label1")),
                  {
                     for case in dataset.iter_mut()
                     {
                        for value in case.expected_outputs.iter_mut()
                        {
                           *value = label0 + (label1 - label0) * *value;
                        }
                     }
                  });

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
