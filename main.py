import click, pathlib
from controller import ControllerBuilder
from pathlib import Path

from lidar_processing import *
import sinks
from dataloader import ROS2Dataloader, ROS2BagDataloader, RLiViTDataloader


builder = None

defaults = {
    "topic-in": "/cloud_all_fields_fullframe"
}


@click.group(chain=True)
def cli():
    global builder
    builder = ControllerBuilder()
@cli.command()
def handle_echo():
    builder.setHandler(Echo())



@cli.command()
@click.option("--threshold", "-t", default=0.2)
@click.option("--query-radius", "-r", default=0.4)
@click.option("--epsilon1", "-e1", default=2.0)
@click.option("--min-samples1", "-m1", default=10)
@click.option("--epsilon2", "-e2", default=1.0)
@click.option("--min-samples2", "-m2", default=10)
@click.option("-s", "--history-size", default=None)
@click.option("--gate", "-g", default=5.0)
@click.option("--rm-ground", "-R", is_flag=True)
@click.option("--boundary-check-radius", "-bcr", default=0.2)
def handle_difference(threshold,
                query_radius, 
                epsilon1, 
                min_samples1, 
                epsilon2, 
                min_samples2, 
                history_size, 
                gate, 
                rm_ground,
                boundary_check_radius):
    tracker = DiffTracker().setThreshold(threshold)\
                         .setQueryRadius(query_radius)\
                         .setDBSCAN1(epsilon1, min_samples1)\
                         .setDBSCAN2(epsilon2, min_samples2)\
                         .setHistorySize(history_size)\
                         .setGate(gate)\
                         .setRemoveGround(rm_ground)\
                         .setBoundaryCheckRadius(boundary_check_radius)
    builder.setHandler(tracker)



@cli.command()
@click.option("--epsilon", "-e", default=0.4)
@click.option("--min-samples", "-m", default=10)
@click.option("-s", "--history-size", default=None)
@click.option("--gate", "-g", default=5)
def handle_baseline( epsilon, min_samples, history_size, gate):
    tracker = Baseline().setDBSCAN(epsilon, min_samples)\
                        .setHistorySize(history_size)\
                        .setGate(gate)
                        
    builder.setHandler(tracker)


### ===========================================================================


@cli.command()
@click.option("--topic", "-t", default=defaults["topic-in"])
def load_ros2(topic):
    dataloader = ROS2Dataloader.builder().setTopicIn(topic)
    builder.setDataloader(dataloader)


@cli.command()
@click.option("--topic", "-t", default=defaults["topic-in"], help="topic to listen to")
@click.option("--input-file", "-i", type=click.Path(exists=True, file_okay=True, readable=True))
def load_ros2bag(topic, input_file):
    input_file = Path(input_file)
    dataloader = ROS2BagDataloader.builder().setTopicIn(topic).setInputFile(input_file)
    builder.setDataloader(dataloader)


@cli.command()
@click.option("--data-path", "-p", type=click.Path(exists=True), default="../../../")
@click.option("--sequence", "-s", type=int, default=0)
@click.option("--wait", "-W", is_flag=True)
def load_rlivit(data_path, sequence, wait):
    print(data_path, type(data_path))
    data_path = pathlib.Path(data_path)
    path = pathlib.Path(data_path / "data/r-livit/R-LiViT_LiDAR/R-LiViT_LiDAR")
    dataloader = RLiViTDataloader.builder().setPathSequence(path.resolve(), [sequence])
    builder.setDataloader(dataloader)


### ===========================================================================


@cli.command()
@click.option("--velocity-threshold", "-vth", default=0.0, help="m/s")
def sink_ros2(velocity_threshold):
    sink = sinks.ROS2PublisherSink()
    builder.setSink(sink)
    builder.setVelocityThresholdVisually(velocity_threshold / 10)


@cli.command()
def sink_ros2bag():
    sink = sinks.ROS2BagSink()
    builder.setSink(sink)


@cli.command()
@click.option("--output-dir", "-o", default="dump")
def sink_file(output_dir):
    sink = sinks.FileSink().setOutputDir(output_dir)
    builder.setSink(sink)


@cli.command()
def sink_mock():
    builder.setSink(sinks.MockSink())

### ===========================================================================


@cli.result_callback()
def runner(*args, **kwargs):
    controller = builder.build()

    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.terminate()


if __name__ == "__main__":
    cli()
    