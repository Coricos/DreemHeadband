# DreemHeadband

## Challenge context

Sleep plays a vital role in an individual’s health and well-being. Sleep progresses in cycles that involve multiple sleep stages : wake, light sleep, deep sleep, rem sleep. Different sleep stages are associated to different physiological functions. Monitoring sleep stage is beneficial for diagnosing sleep disorders. The gold standard to monitor sleep stages relies on a polysomnography study conducted in a hospital or a sleep lab. Different physiological signals are recorded such as electroencephalogram, electrocardiogram etc. Dreem headband allows to monitor such a signal thanks to three kind of sensors : EEG, Pulse Oxymeter and Accelerometer signals. Sleep stage scoring is then performed visually by an expert on epochs of 30 seconds of signals recording. This scoring is both tedious and time-consuming hence prone to error. This is why automatic analysis has gained a lot of interest.

## Challenge goals

Perform sleep stage scoring accurately.

## Data description

Each sample represents 30 seconds of recording for a size total dimension of 22500. There are three kinds of electrophysiological signals: electroencephalogram, pulse oximeter, accelerometer, leading to the following structure of samples in the dataset:

* 4 EEG channels sampled at 125Hz (4 x 125 x 30 = 15000 size per sample)
* 2 Pulse Oxymeter channels (red and infra-red) sampled at 50 Hz (2 x 50 x 30 = 3000 size per sample)
* 3 Accelerometer channels sampled at 50Hz (3 x 50 x 30 = 4500 size per sample)

## Output Description: 

Integer between 0 and 4 representing the sleep stage of the 30-second window. The sleep classification follows the AASM recommendations and was labeled by a single expert.

* 0 : Wake
* 1 : N1 (light sleep)
* 2 : N2
* 3 : N3 (deep sleep)
* 4 : REM (paradoxal sleep)

## Datasets description: 

43830 train samples for 20592 test samples