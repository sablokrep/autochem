use core::error::Error;
use core::result::Result;
use rdkit::{Properties, ROMol};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn regressor_tensor(
    pathfile1: &str,
    pathfile2: &str,
) -> Result<(DenseMatrix<f64>, Vec<i32>), Box<dyn Error>> {
    let smilesfile = File::open(pathfile1).expect("file not present");
    let smilesread = BufReader::new(smilesfile);
    let mut smilesvec: Vec<String> = Vec::new();
    let mut expressionvec: Vec<i32> = Vec::new();
    for i in smilesread.lines() {
        let line = i.expect("line not present");
        let linevec = line.split(",").collect::<Vec<_>>();
        smilesvec.push(linevec[0].to_string());
    }
    let expressionfile = File::open(pathfile2).expect("file not present");
    let expressionread = BufReader::new(expressionfile);
    for i in expressionread.lines() {
        let line = i.expect("file not present");
        let linevec = line.split(",").collect::<Vec<_>>();
        expressionvec.push(linevec[0].parse::<i32>().unwrap());
    }

    let mut smilesvecrmol: Vec<ROMol> = Vec::new();
    for i in smilesvec.iter() {
        let filermol = ROMol::from_smiles(i).unwrap();
        smilesvecrmol.push(filermol);
    }

    let mut smilevec_final: Vec<Vec<f64>> = Vec::new();
    for i in smilesvecrmol.iter() {
        let value = Properties::new().compute_properties(i);
        let finalsmiles = value.iter().map(|(_, y)| y).cloned().collect::<Vec<f64>>();
        smilevec_final.push(finalsmiles);
    }

    let densematrix: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&smilevec_final).unwrap();

    Ok((densematrix, expressionvec))
}
