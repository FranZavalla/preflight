use clap::Parser;
use miette::Result;

#[derive(Parser)]
#[command(
    name = "preflight",
    about = "Aiken smart contract vulnerability auditor",
    version
)]
struct Cli {
    #[command(flatten)]
    args: preflight::Args,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    preflight::run(cli.args)
}
