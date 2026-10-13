proc export data=sashelp.heart
  outfile="/workspaces/myfolder/heart_risk_project/data/raw/heart_raw.csv"
  dbms=csv replace;
run;