mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
use figlet_rs::FIGfont;
use smartcore::linear::logistic_regression::LogisticRegression;
mod smile;
use crate::smile::readsmiles;
use crate::smile::readsmiles_predict;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::fs::File;
use std::io::Write;
mod knn;
use crate::knn::classsification_knn;
mod reger;
mod regrssor;
use crate::reger::reger_add;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("AutoChem");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::SmileClassify {
            smiles,
            expression,
            threads,
            threshold,
            predexp,
            predsmiles,
        } => {
            let n_threads = threads.parse::<usize>().unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("file not present");
            pool.install(|| {
                let valueinput = readsmiles(smiles, expression, threshold).unwrap();
                let logisticvalue =
                    LogisticRegression::fit(&valueinput.0, &valueinput.1, Default::default())
                        .unwrap();
                let predvalue = readsmiles_predict(predsmiles, predexp).unwrap();
                let valuepred = logisticvalue.predict(&predvalue).unwrap();
                let accuracy_value = accuracy(&valueinput.1, &valuepred);
                println!(
                    "The accuracy value on the entire training datasets is {}",
                    accuracy_value
                );

                let splitratio =
                    train_test_split(&valueinput.0, &valueinput.1, 0.2, true, Some(2811));
                let logistictest =
                    LogisticRegression::fit(&splitratio.0, &splitratio.2, Default::default())
                        .unwrap();
                let logisticpredict = logistictest.predict(&splitratio.1).unwrap();
                let accuracytestsplit = accuracy(&splitratio.3, &logisticpredict);
                println!(
                    "The  accuracy of the predicted values on the split basis is: {}",
                    accuracytestsplit
                );
                let splitpredict = logistictest.predict(&predvalue).unwrap();
                let mut filewrite = File::create("splitpredict.txt").expect("file not present");
                for i in splitpredict.iter() {
                    writeln!(filewrite, "{}", i).expect("line not present");
                }
            });

            let knnclassify =
                classsification_knn(smiles, expression, threshold, predsmiles, predexp).unwrap();
            println!("The KNN classifer has finished:{}", knnclassify);
        }
        Commands::SmileRegressor {
            smiles,
            expression,
            threads,
            predexp,
            predsmiles,
        } => {
            let threadlaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadlaunch)
                .build()
                .expect("threads not present");
            value.install(|| {
                let valueunpack = reger_add(smiles, expression, predexp, predsmiles).unwrap();
                print!("The regressor has finished:{:?}", valueunpack);
            });
        }
    }
}
