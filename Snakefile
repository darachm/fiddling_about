
rule fit_nn:
  input: 
    flow_csv="data/fl_cn_training_data.csv",
  output: 
    scores_csv="scores_trained_on_all.csv",
  shell:
    """
    ./maudz.py
    """

rule make_plots:
  input: 
    scores_csv="scores_trained_on_all.csv",
  output: 
    "importance_of_variables.jpg"
  shell:
    """
    Rscript plotz.R
    """

rule zipper:
  input: "data/"
  output: "the_data.zip"
  shell: "zip -r {output} {input}"

rule unzipper:
  input: "the_data.zip"
  output: "data/"
  shell: "unzip {input}"
