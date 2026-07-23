import toml, click, pathlib
from controller import ControllerBuilder
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

import dataloader, lidar_processing, sinks

class Runner:
    def __init__(self, parameter_file):
        data = toml.load(parameter_file)

        defaults_init = {"dataloader":{},
                         "handler":{},
                         "sink":{"method": "MockSink"}}
        
        defaults = {**defaults_init, **data["defaults"]}

        self.runs = []
        for i, run in enumerate(data["run"]):
            builder = ControllerBuilder()
            run = {**defaults, **run}
            name = run["name"]
            del run["name"]
            run = {k:{**defaults[k], **run[k]} for k in defaults}
            pprint(run)
            
            builder.setName(name)
            dl = run["dataloader"]
            
            loader = getattr(dataloader, dl["method"]).builder()
            loader.fromParameters(**dl)
            builder.setDataloader(loader)

            hd = run["handler"]
            
            handler = getattr(lidar_processing, hd["method"])()
            handler.fromParameters(**{**vars(handler), **hd})
            builder.setHandler(handler.finish())

            sk = run["sink"]
            
            sink = getattr(sinks, sk["method"])()
            sink.fromParameters(**{**vars(sink), **sk})
            builder.setSink(sink)

            ro = run["results"]
            builder.setResultsHandlerPath(pathlib.Path(ro["path"]), i)

            self.runs.append(builder.build())


    def run(self):
        for cntrl in self.runs:
            try:
                cntrl.run()
            except KeyboardInterrupt:
                return
            finally:
                cntrl.terminate()


    def run_parallel(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(c.run) for c in self.runs]

            try:
                for f in as_completed(futures):
                    f.result()
            except KeyboardInterrupt:
                pass
            finally:
                for cntrl in self.runs:
                    cntrl.terminate()


def run_(cntrl):
    try:
        cntrl.run()
    except KeyboardInterrupt:
        return
    finally:
        cntrl.terminate()


@click.command()
@click.argument("file")
def cli(file):
    print(file)
    runner = Runner(file)

    runner.run()



if __name__ == "__main__":
    cli()