use crate::regrssor::regressor_tensor;
use crate::smile::readsmiles_predict;
use core::error::Error;
use core::option::Option::Some;
use core::result::Result;
use core::writeln;
use smartcore::ensemble::extra_trees_regressor::ExtraTreesRegressor;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{mean_absolute_error, mean_squared_error, r2};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use std::fs::File;
use std::io::Write;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn reger_add(
    pathstring1: &str,
    pathstring2: &str,
    path3: &str,
    path4: &str,
) -> Result<String, Box<dyn Error>> {
    let tensorunpack = regressor_tensor(pathstring1, pathstring2).unwrap();
    let predictunpack = readsmiles_predict(path3, path4).unwrap();
    let trainsplit = train_test_split(&tensorunpack.0, &tensorunpack.1, 0.8, true, Some(2811));
    let regressor_linear =
        LinearRegression::fit(&trainsplit.0, &trainsplit.2, Default::default()).unwrap();
    let predictvalue = regressor_linear.predict(&trainsplit.1).unwrap();
    let meansquare_linear = mean_squared_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let meanabsolute_linear = mean_absolute_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let r2_linear = r2(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    println!(
        "The value for the linear regressor: {}\t{}\t{}",
        meansquare_linear, meanabsolute_linear, r2_linear
    );
    let predict_change = regressor_linear.predict(&predictunpack).unwrap();
    let mut filelinear = File::create("linear_reg.txt").expect("file not present");
    for i in predict_change.iter() {
        writeln!(filelinear, "{:?}\n", i).expect("fline not present");
    }

    let extra_reg =
        ExtraTreesRegressor::fit(&trainsplit.0, &trainsplit.2, Default::default()).unwrap();
    let predictvalue_extra = extra_reg.predict(&trainsplit.1).unwrap();
    let meansquare_extra = mean_squared_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_extra
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let meanabsolute_extra = mean_absolute_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_extra
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let r2_extra_reg = r2(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_extra
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );

    println!(
        "The value for the extra_tree regressor: {}\t{}\t{}",
        meansquare_extra, meanabsolute_extra, r2_extra_reg
    );

    let actualpredict = extra_reg.predict(&predictunpack);

    let mut extra_write = File::create("extra_pred.txt").expect("file not present");
    for i in actualpredict.iter() {
        writeln!(extra_write, "{:?}\n", i).expect("file not present");
    }

    let random_regressor =
        RandomForestRegressor::fit(&trainsplit.0, &trainsplit.2, Default::default()).unwrap();

    let predictvalue_random = random_regressor.predict(&trainsplit.1).unwrap();
    let meansquare_random = mean_squared_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_random
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let meanabsolute_random = mean_absolute_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_random
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let r2_random = r2(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_extra
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );

    let actual_random = random_regressor.predict(&predictunpack).unwrap();

    println!(
        "The value for the random regressor: {}\t{}\t{}",
        meansquare_random, meanabsolute_random, r2_random
    );

    let mut filerandom = File::create("random_file.txt").expect("file not present");
    for i in actual_random.iter() {
        writeln!(filerandom, "{}\n", i).expect("file not present");
    }

    let knn_reg = KNNRegressor::fit(&trainsplit.0, &trainsplit.2, Default::default()).unwrap();
    let predictvalue_knn = knn_reg.predict(&trainsplit.1).unwrap();
    let meansquare_knn = mean_squared_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_knn
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let meanabsolute_knn = mean_absolute_error(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_knn
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );
    let r2_value_knn = r2(
        &trainsplit
            .3
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
        &predictvalue_extra
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    );

    println!(
        "The value for the knn regressor: {}\t{}\t{}",
        meansquare_knn, meanabsolute_knn, r2_value_knn
    );

    let actual_knn = knn_reg.predict(&predictunpack).unwrap();
    let mut knn_write = File::create("actual_knn.txt").expect("file not present");
    for i in actual_knn.iter() {
        writeln!(knn_write, "{}\n", i).expect("file not present");
    }

    Ok("The value has been done".to_string())
}
