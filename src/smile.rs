use rdkit::{Properties, ROMol};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com
 */

pub fn readsmiles(
    pathfile: &str,
    genexpression: &str,
    threshold: &str,
) -> Result<(DenseMatrix<f64>, Vec<i32>), Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut smilesvec: Vec<String> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let valueadd = line.split(",").collect::<Vec<_>>();
        smilesvec.push(valueadd[4].to_string());
    }
    let mut valuesmiles: Vec<Vec<f64>> = Vec::new();
    for i in smilesvec.iter() {
        let smileparse = ROMol::from_smiles(i).unwrap();
        let properties: HashMap<String, f64> = Properties::new().compute_properties(&smileparse);
        let mut values: Vec<f64> = Vec::new();
        for (_, val) in properties.iter() {
            values.push(*val);
        }
        valuesmiles.push(values);
    }
    let valueexpression = expression(genexpression).unwrap();
    let mut classseq: Vec<i32> = Vec::new();
    for i in valueexpression.iter() {
        if i.to_string().parse::<i32>().unwrap() < threshold.to_string().parse::<i32>().unwrap() {
            classseq.push(0)
        } else if i.to_string().parse::<i32>().unwrap()
            > threshold.to_string().parse::<i32>().unwrap()
        {
            classseq.push(1)
        }
    }

    let mut combinedvecfinal: Vec<Vec<f64>> = Vec::new();
    for i in valueexpression.iter() {
        for smile in valuesmiles.iter() {
            let mut veca = smile.clone();
            veca.push(*i);
            combinedvecfinal.push(veca);
        }
    }

    let finaldesnsematrix = DenseMatrix::from_2d_vec(&combinedvecfinal).unwrap();
    Ok((finaldesnsematrix, classseq))
}

pub fn expression(pathfile: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut expressionvec: Vec<f64> = Vec::new();
    for i in fileread.lines() {
        let value = i.expect("file not present");
        expressionvec.push(value.parse::<f64>().unwrap());
    }
    Ok(expressionvec)
}

pub fn readsmiles_predict(
    pathfile: &str,
    genexpression: &str,
) -> Result<DenseMatrix<f64>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut smilesvec: Vec<String> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let valueadd = line.split(",").collect::<Vec<_>>();
        smilesvec.push(valueadd[4].to_string());
    }
    let mut valuesmiles: Vec<Vec<f64>> = Vec::new();
    for i in smilesvec.iter() {
        let smileparse = ROMol::from_smiles(i).unwrap();
        let properties: HashMap<String, f64> = Properties::new().compute_properties(&smileparse);
        let mut values: Vec<f64> = Vec::new();
        for (_, val) in properties.iter() {
            values.push(*val);
        }
        valuesmiles.push(values);
    }

    let valueexpression = expression(genexpression).unwrap();
    let mut combinedvecfinal: Vec<Vec<f64>> = Vec::new();

    for i in valueexpression.iter() {
        for smile in valuesmiles.iter() {
            let mut veca = smile.clone();
            veca.push(*i);
            combinedvecfinal.push(veca);
        }
    }

    let finaldesnsematrix = DenseMatrix::from_2d_vec(&combinedvecfinal).unwrap();
    Ok(finaldesnsematrix)
}
