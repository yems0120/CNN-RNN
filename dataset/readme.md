## Sample download

- The N-CMAPSS dataset is publicly available from NASA’s Prognostics Center of Excellence Data Set Repository, accessible for download from the following link: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository under the heading “17. Turbofan Engine Degradation Simulation-2”.

  ## Sample creator

- Creating training/test sample as follows: https://github.com/mohyunho/N-CMAPSS_DL

  Then, you can get npz files for each of 10 engines by running the python codes below.

  ```
  python3 sample_creator_unit_auto.py -w 50 -s 50 --test 0 --sampling 10
  ```

  

