use std::{io, io::Write};
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};

fn main() {
    let qa_model = QuestionAnsweringModel::new(Default::default()).unwrap();
    let mut question = String::new();
    let mut context = String::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("Context : ");
        stdout.flush().unwrap();
        stdin.read_line(&mut context).unwrap();
        context = context.trim().to_string();

        print!("Question: ");
        stdout.flush().unwrap();
        stdin.read_line(&mut question).unwrap();
        question = question.trim().to_string();

        let answers = qa_model.predict(
            &[QaInput { question: question.clone(), context: context.clone() }],
            1, 32);

        println!("Answers : {answers:?}");        
    }
}
