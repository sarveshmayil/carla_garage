@echo off
setlocal enabledelayedexpansion

cd ..
mkdir data
cd data

:down_load_unzip
set "scenario=%1"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects-2/jaeger2023arxiv/dataset/!scenario!.zip"
tar -xf "!scenario!.zip" 2>nul
del "!scenario!.zip"

for %%A in (
  ll_dataset_2023_05_10
  ::rr_dataset_2023_05_10
  ::lr_dataset_2023_05_10
  ::rl_dataset_2023_05_10
  ::s1_dataset_2023_05_10
  ::s3_dataset_2023_05_10
  ::s7_dataset_2023_05_10
  ::s10_dataset_2023_05_10
  ::s4_dataset_2023_05_10
  ::s8_dataset_2023_05_10
  ::s9_dataset_2023_05_10
) do (
  call :down_load_unzip "%%A"
)

endlocal