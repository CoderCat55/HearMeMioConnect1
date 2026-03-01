
 * check calibration buffer for size and if its getting data from true source in a true way.

 * Folders:
   * Rawdata: P1, P2, P3, P4
   * Rawdataplots: P1, P2, P3, P4
   * Prodata (Processed data): P1, P2, P3, P4
   * Prodataplot (processed data plots): P1, P2, P3

 * Data handler fix:
   * Datas will be written to shared memory when sEMG comes
   * sEMG data will be written taking mean average of them (assuming emg sends 100hz for both values) -> or find another way.
 * check expected hz in the program and data handler

Done and should be tested
* Delete old SEM datas and folders and plots
 * Calibration_duration = 5 seconds
 * Disable countdown in calibration

 * new general calibrationcalibration method:
    *user will write gcb and after the program will ask enter the values:
    user answer contains "subfolder gesturename" they are seperated from space chrachters -> set folder and gesture name. Then 
    * when r pressed start recording.
    * when x pressed escape calibration loop

    until x is pressed:
                    if r is pressed call the funciton and save the datas
                    else wait for user to enter something

get_recent_data_from_shared_mem kullanılmıyor neden var?
her modelin trainleri ayrı ayrı olsun