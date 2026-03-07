use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "autochem",
    version = "1.0",
    about = "Autochemical ML
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// logistic classifier for Smiles
    SmileClassify {
        /// path to the smiles chmeical uses
        smiles: String,
        /// path to the expression file
        expression: String,
        /// threads for the analysis
        threads: String,
        /// expression threashold for the classification
        threshold: String,
        /// prediction expression file
        predexp: String,
        /// pred smiles file
        predsmiles: String,
    },
    ///  regressor on smiles
    SmileRegressor {
        /// path to the smile chemical uses
        smiles: String,
        /// path to the expression file
        expression: String,
        /// threads for the analysis
        threads: String,
        /// prediction expression file
        predexp: String,
        /// pred smile file
        predsmiles: String,
    },
}
