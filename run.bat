@echo off
@REM Run all 4 algorithms (none, gosine_block, pure_brs, hybrid_brs_block) across all 3 bandit policies.
@REM Using --no_show saves each run's .npz archive and .png plots without blocking on plot windows.
@REM Optional usage: .\run.bat <subfolder_name> (saves inside results\<subfolder_name>\)

set SUBFOLDER_ARG=
if not "%~1"=="" set SUBFOLDER_ARG=--subfolder "%~1"

@REM --- Algorithm 0: Unprotected GosInE (Baseline) ---
python main.py -k 100 -t 1000000 --policy Thompson-Sampling --defense none --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy Epsilon-Greedy --defense none --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy UCB --defense none --no_show %SUBFOLDER_ARG%

@REM --- Algorithm 1: GosInE-Block (Vial et al., 2021) ---
python main.py -k 100 -t 1000000 --policy Thompson-Sampling --defense gosine_block --eta 2.0 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy Epsilon-Greedy --defense gosine_block --eta 2.0 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy UCB --defense gosine_block --eta 2.0 --no_show %SUBFOLDER_ARG%

@REM --- Algorithm 2: Pure BRS-GosInE (Josang & Ismail, 2002 adapted) ---
python main.py -k 100 -t 1000000 --policy Thompson-Sampling --defense pure_brs --lam 0.85 --theta_min -0.50 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy Epsilon-Greedy --defense pure_brs --lam 0.85 --theta_min -0.50 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy UCB --defense pure_brs --lam 0.85 --theta_min -0.50 --no_show %SUBFOLDER_ARG%

@REM --- Algorithm 3: Hybrid BRS-Block (Reputation-Scaled Escalating Blocking) ---
python main.py -k 100 -t 1000000 --policy Thompson-Sampling --defense hybrid_brs_block --eta 2.0 --lam 0.85 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy Epsilon-Greedy --defense hybrid_brs_block --eta 2.0 --lam 0.85 --no_show %SUBFOLDER_ARG%
python main.py -k 100 -t 1000000 --policy UCB --defense hybrid_brs_block --eta 2.0 --lam 0.85 --no_show %SUBFOLDER_ARG%

echo Run finished! All archives and plots have been saved.