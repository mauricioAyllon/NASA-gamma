"""Read list mode data

The file provides a python based list mode data parser.

The input can either come from a data stream or from one or several
binary files (in case a large binary data set is split across multiple
files).

We keep all the data in numpy arrays, whic are either memmaped or
pre-allocated, to make data access fast.

"""

from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

try:
    import cbitstruct as bitstruct
except ModuleNotFoundError:
    import bitstruct

import msgspec

# from rich import print


class Event(msgspec.Struct):
    """Data storage for a single event that has attribute access.

    This is similar to a namedtuple, but the msgspec implemenation is faster.
    """

    channel: int = -1
    crate: int = -1
    slot: int = -1
    timestamp: float = -1.0
    CFD_fraction: int = -1
    energy: int = -1
    trace: int = -1
    CFD_error: int = -1
    pileup: int = -1
    trace_flag: int = -1
    Esum_trailing: int = -1
    Esum_leading: int = -1
    Esum_gap: int = -1
    baseline: float = -1.0
    QDCSum0: int = -1
    QDCSum1: int = -1
    QDCSum2: int = -1
    QDCSum3: int = -1
    QDCSum4: int = -1
    QDCSum5: int = -1
    QDCSum6: int = -1
    QDCSum7: int = -1
    ext_timestamp: float = -1.0
    chunk_timestamp: float = -1  # not a pixie16 field

    def to_list(self):
        return (getattr(self, f) for f in self.__struct_fields__)


# The bit structure to unpack the pixie binary format
EVENTS_PARTS = {
    "header": (
        "b1u14u5u4u4u4u32u3u13u16b1u15u16",
        (
            "pileup",
            "Event Length",
            "Header Length",
            "crate",
            "slot",
            "channel",
            "EVTTIME_LO",
            "CFD trigger source bits",
            "CFD Fractional Time",
            "EVTTIME_HI",
            "trace_flag",
            "Trace Length",
            "energy",
        ),
    ),
    "energy": (
        "u32" * 3 + "f32",
        (
            "Esum_trailing",
            "Esum_leading",
            "Esum_gap",
            "baseline",
        ),
    ),
    "Qsums": (
        "u32" * 8,
        (
            "QDCSum0",
            "QDCSum1",
            "QDCSum2",
            "QDCSum3",
            "QDCSum4",
            "QDCSum5",
            "QDCSum6",
            "QDCSum7",
        ),
    ),
    "Ext Time": (
        "u32p16u16",
        (
            "Ext_TS_Lo",
            "Ext_TS_Hi",
        ),
    ),
}

# We now build all possible combinations out of these parts that can
# show up in list mode data We also add a "0" option to just read the
# first row, since this row contains the header length we need to read
HEADER_FORMATS = bitstruct.CompiledFormatDict(
    EVENTS_PARTS["header"][0], EVENTS_PARTS["header"][1]
)

# when more information is in the header, we look it up using the
# following data structure. The key into this dictionary is how many
# 32-bit words we still need to read
EXTRA_HEADER_FORMATS = {
    2: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["Ext Time"][0],
        EVENTS_PARTS["Ext Time"][1],
    ),
    4: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["energy"][0],
        EVENTS_PARTS["energy"][1],
    ),
    8: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["Qsums"][0],
        EVENTS_PARTS["Qsums"][1],
    ),
    12: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["energy"][0] + EVENTS_PARTS["Qsums"][0],
        EVENTS_PARTS["energy"][1] + EVENTS_PARTS["Qsums"][1],
    ),
    6: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["energy"][0] + EVENTS_PARTS["Ext Time"][0],
        EVENTS_PARTS["energy"][1] + EVENTS_PARTS["Ext Time"][1],
    ),
    10: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["Qsums"][0] + EVENTS_PARTS["Ext Time"][0],
        EVENTS_PARTS["Qsums"][1] + EVENTS_PARTS["Ext Time"][1],
    ),
    14: bitstruct.CompiledFormatDict(
        EVENTS_PARTS["energy"][0]
        + EVENTS_PARTS["Qsums"][0]
        + EVENTS_PARTS["Ext Time"][0],
        EVENTS_PARTS["energy"][1]
        + EVENTS_PARTS["Qsums"][1]
        + EVENTS_PARTS["Ext Time"][1],
    ),
}


class EmptyError(Exception):
    """We reached the end of the last file."""


class LeftoverBytesError(Exception):
    """To be raised if a partial event is left in the byte stream/file."""


class FileReader:
    """Read binary data from multiple files.

    Use mmap for fast data access to binary data on disc. The data can
    be distributed across several files.

    This class should  be used as a context manager:
    >>> files = ("file1.bin", "file2.bin")
    >>> with FileReader(files) as r:
    >>>     r.read(100)

    or one needs to call self.open_files() and self.close_files()
    manually.

    """

    def __init__(self, files: Iterable[Path]):
        self.current_file = 0  # which file do we need to read from
        self.files = files  # list of all files
        self.file_size = []
        self.position = 0  # relative to the current file
        self.mmap = []
        self.done = False  # have we reached the end of the last file?

    def open_files(self):
        for f in self.files:
            self.mmap.append(np.memmap(f, dtype=np.uint32, mode="r"))
            self.file_size.append(len(self.mmap[-1]))

    def close_files(self):
        self.file_size = []
        self.file_objects = []
        self.mmap = []
        self.current_file = 0

    def __enter__(self):
        """Opens all files, check their sizes and open a mmap read_only connection."""
        self.open_files()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes all file descriptors and mmaps at the end of the context manager."""
        self.close_files()

    def advance(self, size: int):
        """Not needed, since all the data is in the files and we can
        just advance the file position during read()"""
        pass

    def read(self, size: int, offset: int = 0) -> bytes:
        """Retrieve the next `size` bytes from the files and advance the file position.

        Parameters
        ----------
        size
            How many bytes should be returned.
        offset
            ignored here (used in Streamreader)

        Raises
        ------
        EmptyError
            When data of the requested size is not available in the data stream.

        """

        if self.done:
            raise EmptyError()

        end = size + self.position
        if end <= self.file_size[self.current_file]:
            out = self.mmap[self.current_file][self.position : end]
            self.position += size
            return out

        tmp = self.mmap[self.current_file][self.position :]
        to_read = size - len(tmp)

        self.current_file += 1
        self.position = 0

        if self.current_file == len(self.files):
            self.done = True
            if len(tmp) == 0:
                raise EmptyError
            raise LeftoverBytesError

        rest = self.read(to_read)
        if tmp.size > 0:
            return np.concatenate((tmp, rest))
        else:
            return rest


class StreamReader:
    """Handle streamed binary data.

    Provides the same interface as FileReader and can be used as a direct replacement.

    New data can be added using the `put` method.

    Internally we store the data in a np.array of np.uint32.

    We make the array large enough to hold the data and have the
    buffer grow and shrink as needed.

    This is similar to a ring buffer, but we just allocate twice the buffer
    size memory and then keep track of the left and right position in
    the buffer and shift the data to the far left every now and then.

    """

    def __init__(self, initial_buffer_size=25_000_000) -> None:
        self.buffer_size = initial_buffer_size
        self.buffer_size_orig = self.buffer_size
        self.buffer = np.empty(2 * self.buffer_size, dtype=np.uint32)
        self.left = 0
        self.right = 0

    def put(self, input: np.ndarray) -> None:
        """Add data to the buffer.

        Also increase/decrease the buffer size as needed.
        """

        L = input.size
        current = self.right - self.left

        buffer_size_change = False
        # adjust buffer so that it will be large enough for the new data
        while current + L >= self.buffer_size:
            self.buffer_size *= 2
            buffer_size_change = True

        # Can we shrink the buffer?
        while (self.buffer_size > self.buffer_size_orig) and (
            current + L < self.buffer_size // 2
        ):
            self.buffer_size = self.buffer_size // 2
            buffer_size_change = True

        # check if we need to shift the buffer
        if self.left > self.buffer_size or buffer_size_change:
            self.adjust_and_shift_buffer()

        # add input
        self.buffer[self.right : self.right + L] = input
        self.right += L

    def adjust_and_shift_buffer(self) -> None:
        """Create a new buffer and copy the old buffer to beginning.

        This also takes care of changing the size of the buffer
        """
        if (self.left == 0) and (self.buffer.size == self.buffer_size):
            return

        current = self.right - self.left
        tmp = np.empty(2 * self.buffer_size, dtype=np.uint32)
        tmp[:current] = self.buffer[self.left : self.right]
        self.buffer = tmp
        self.left = 0
        self.right = current

    def read(self, size: int, offset: int = 0) -> np.ndarray:
        """Return `size` 32-bit words from the buffer as a numpy array.

        In the streamreader, there might be an incomplete event at the end of the buffer.
        However, we have several read statements to read a single event. We cannot advance
        self.left directly, unless we read a whole event (up to 3 read()). `offset` is used to
        read partial data and advance() is used to move self.left once a full event is read.
        """
        current = self.right - (self.left + offset)
        if size <= current:
            out = self.buffer[self.left + offset : self.left + offset + size]
            return out
        raise EmptyError

    def advance(self, size: int) -> None:
        """Advance self.left once a full event is read (possible after multiple read statement)"""
        self.left += size

    def __len__(self):
        """Return the amount of data in the buffer."""
        return self.right - self.left


class ListModeDataReader:
    def __init__(self, reader: FileReader | StreamReader):
        self.reader = reader

    def pop(self) -> Event:
        """Pop the top event from queue"""

        # try to get all the words for the next event
        # if the complete next event isn't in queue an exception is thrown but no data is discarded
        event_header = HEADER_FORMATS.unpack(self.reader.read(4).byteswap().tobytes())
        header_length = event_header["Header Length"]
        extra_header_length = header_length - 4
        event_length = event_header["Event Length"]

        # it's now save to read both header and trace
        if extra_header_length > 0:
            header_words = self.reader.read(extra_header_length, 4).byteswap().tobytes()
            # unpack words/trace into their field values
            try:
                event_values = EXTRA_HEADER_FORMATS[extra_header_length].unpack(
                    header_words
                )
                event_header.update(**event_values)
            except KeyError:
                print("Something went wrong with the binary data.", flush=True)
                raise

        trace_length = event_length - header_length
        if trace_length:
            trace_data = self.reader.read(trace_length, header_length).tobytes()
            trace = np.frombuffer(trace_data, f"({trace_length},2)<u2", count=1)[
                0, :, :
            ]
            trace = trace.flatten()
        else:
            trace_data = None
            trace = ()

        # if reader is a filereader, reader.read() will already advance the positions
        # in streamreader, we might still be waiting for more data from the pixie
        # The workaround is to only remove event once all read operation worked and
        # have read() not advance the buffer location.
        self.reader.advance(event_length)

        # calculate event values that need to be derived from field values
        CFD_bits = event_header.pop("CFD trigger source bits")
        CFD_error = CFD_bits == 7
        CFD_fractional_time = event_header.pop("CFD Fractional Time")
        if CFD_bits == 7:
            CFD_fraction = 0
        else:
            CFD_fraction = ((CFD_bits - 1) + CFD_fractional_time / 8192) * 2

        timestamp = (
            event_header.pop("EVTTIME_LO") + event_header.pop("EVTTIME_HI") * 2**32
        ) * 10

        if "Ext_TS_Lo" in event_header:
            ext_timestamp = (
                event_header.pop("Ext_TS_Lo") + event_header.pop("Ext_TS_Hi") * 2**32
            )

            event_header.update({"ext_timestamp": ext_timestamp})

        for key in ["Event Length", "Header Length", "Trace Length"]:
            event_header.pop(key, None)

        event_header.update(
            {
                "timestamp": timestamp,
                "CFD_fraction": CFD_fraction,
                "trace": trace,
                "CFD_error": CFD_error,
            }
        )

        return Event(**event_header)

    def iterevents(self):
        """Will raise an exception to exit loop."""
        while True:
            yield self.pop()

    def pop_all(self):
        out = []
        while True:
            try:
                event = self.pop()
            except (EmptyError, LeftoverBytesError):
                break
            out.append(event)
        return out


def events_from_files_generator(files):
    """Yields events from a list of binary files"""
    try:
        with FileReader(files) as r:
            reader = ListModeDataReader(r)
            while not reader.reader.done:
                yield from reader.iterevents()
    except (EmptyError, LeftoverBytesError):
        return


def to_tuple(obj):
    if isinstance(obj, str | Path):
        obj = [obj]
    else:
        try:
            obj = tuple(obj)
        except TypeError:
            obj = obj
    return obj


def read_list_mode_data(files, buffer_size=int(1e9)):
    """Loads a list of binary files into a pandas DataFrame"""
    files = to_tuple(files)
    events = (e.to_list() for e in events_from_files_generator(files))
    return pd.DataFrame.from_records(events, columns=Event.__struct_fields__)


def read_list_mode_data_as_events(files, buffer_size=int(1e9), max_size=None):
    """Returns a list of at most max_size events from files"""
    files = to_tuple(files)
    gen = events_from_files_generator(files)
    return list(islice(gen, max_size))


def sort_events_by_channel(events, channel_list=None):
    """Sort events by channel.

    Takes the output of `read_list_mode_data_as_events` and sorts them
    into a dictionary that has the channel number as key and as value
    a list of events (still including the channel number).

    Parameters
    ----------
    events
        List of events as returned by `read_list_mode_data_as_events`
    channel_list
        Optional list of channels to include. This can help to reduce
        the amount of data.

    """

    out = defaultdict(list)

    for e in events:
        ch = e.channel
        if channel_list is not None and ch in channel_list:
            out[ch].append(e)
        else:
            out[ch].append(e)

    return out
