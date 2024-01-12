from io import BufferedWriter, RawIOBase, DEFAULT_BUFFER_SIZE
from typing import Union


class BufferedRingWriter(BufferedWriter):
    @property
    def header_size(self) -> int:
        return self._header_size

    @property
    def ring_size(self) -> int:
        return self._ring_size

    @property
    def seam_position(self) -> int:
        return (self._virtual_file_size - self._header_size) % self._ring_size

    def __init__(self, raw: RawIOBase, ring_size: int, buffer_size: int = DEFAULT_BUFFER_SIZE,
                 header_size: int = 0):
        self._unwrapped_pos = 0
        self._header_size = header_size
        self._ring_size = ring_size

        super().__init__(raw, buffer_size)

        orig_pos = super().tell()
        self._virtual_file_size = super().seek(0, 2)
        super().seek(min(orig_pos, self._header_size + self._ring_size))
        super().truncate(self._header_size + self._ring_size)

    def _write_and_advance(self, b: Union[bytes, bytearray]) -> int:
        written = super().write(b)
        self._unwrapped_pos += written
        self._virtual_file_size = max(self._virtual_file_size, self._unwrapped_pos)

        return written

    def wrap_pos(self, pos: int):
        if pos < self._header_size:
            return pos
        else:
            return ((pos - self._header_size) % self._ring_size) + self._header_size

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            super().seek(self.wrap_pos(offset))
            self._unwrapped_pos = offset
        elif whence == 1:
            super().seek(self.wrap_pos(self._unwrapped_pos + offset))
            self._unwrapped_pos = self._unwrapped_pos + offset
        elif whence == 2:
            super().seek(self.wrap_pos(self._virtual_file_size + offset))
            self._unwrapped_pos = self._virtual_file_size + offset

        return self._unwrapped_pos

    def tell(self) -> int:
        return self._unwrapped_pos

    def write(self, b: Union[bytes, bytearray]) -> int:
        written = 0

        if self._unwrapped_pos < self._header_size:
            header_rem = self._header_size - self._unwrapped_pos

            if header_rem < len(b):
                raise IOError("write operation in header exceeds header bounds")
            else:
                written += self._write_and_advance(b)
        elif self._unwrapped_pos < self._virtual_file_size - self._ring_size:
            return 0  # Write area in limbo
        else:
            ring_pos = super().tell() - self._header_size
            ring_rem = self._ring_size - ring_pos
            while written < len(b):
                written += self._write_and_advance(b[written:written + ring_rem])

                self.seek(self._unwrapped_pos)  # wrap back to ring start if necessary.
                ring_pos = super().tell() - self._header_size
                ring_rem = self._ring_size - ring_pos

        return written
