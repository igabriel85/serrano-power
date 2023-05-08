from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain
from pyJoules.energy_meter import EnergyMeter
import pandas as pd

import time

domains = [RaplPackageDomain(0), RaplCoreDomain(0), RaplDramDomain(0), ]
devices = DeviceFactory.create_devices(domains)
meter = EnergyMeter(devices)

def foo():
    time.sleep(5)

def bar():
    time.sleep(5)

traces = []
for i in range(5):
    meter.start(tag=f'foo_{i}')
    foo()
    meter.record(tag=f'bar_{i}')
    bar()
    meter.stop()
    trace = meter.get_trace()
    traces.append(trace)


list_time = []
list_tags = []
list_durations = []
list_energys = []
for trace in traces:
    for e in trace:
            print(e.timestamp, e.tag, e.duration, e.energy)
            list_time.append(e.timestamp)
            list_tags.append(e.tag)
            list_durations.append(e.duration)
            list_energys.append(e.energy['package_0'])

eng_rep = {
    'timestamp': list_time,
    'tag': list_tags,
    'duration': list_durations,
    'energy' : list_energys
}
df_report = pd.Dataframe(eng_rep)
df_report.to_csv('df_rep.csv')