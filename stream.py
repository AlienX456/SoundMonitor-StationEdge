from json import dumps
import logging
import os
import uuid
from datetime import datetime
from dateutil import tz
from kafka import KafkaProducer
from inferencer.YAMnet import YAMnet
from nivelDeRuido.nivelRuido import NivelRuido
from resources.recorder import recorder
from resources.deviceInfo import deviceInfo

logging.getLogger().setLevel(logging.INFO)

device = deviceInfo()
info = device.getInfoObj()
inferencer = YAMnet()
nivel_ruido = NivelRuido()
recorder = recorder(dirname='./', time=10)
inferencer_identifier = uuid.uuid4().__str__()

producer = KafkaProducer(bootstrap_servers=[os.environ["KAFKA_BOOTSTRAP_SERVER_ONE"]],
                         value_serializer=lambda x: dumps(x).encode("utf-8"))

try:

    while True:

        filename = "raspberry-" + uuid.uuid1().__str__()
        ruta = recorder.record(id=filename)
        now = datetime.now(tz=tz.tzutc())
        date_time = now.strftime("%Y-%m-%dT%H:%M:%S")
        info['audio_uuid'] = filename
        info['time'] = date_time

        inferencer_result = inferencer.run_inferencer(filename)
        Leq = nivel_ruido.calcular_db(filename)
        os.remove(filename)


        dataToSend = {'device_info': info, 'inference_result': inferencer_result,
                      "inferencer_name": 'YAMNET', "noise_level": Leq}
        logging.info("Sending result :{} to process_result_event".format(dataToSend))
        producer.send("process_result_event", value=dataToSend)


        logging.info("{} Jobs Finished".format(filename))


except Exception as e:
    logging.error('There was an error while Connecting: {}'.format(str(e)))
