"""Efficient serialization using msgpack for network messages"""
import msgpack
import zlib
from typing import Any, Union
import json


class NetworkSerializer:
    """Handles efficient serialization for P2P messages"""
    
    def __init__(self, compress_threshold: int = 1024):
        """
        Initialize serializer
        
        Args:
            compress_threshold: Compress messages larger than this (bytes)
        """
        self.compress_threshold = compress_threshold
        
    def serialize(self, data: Any, compress: bool = True) -> bytes:
        """
        Serialize data to bytes using msgpack
        
        Args:
            data: Data to serialize
            compress: Whether to compress large messages
            
        Returns:
            Serialized bytes
        """
        # Convert to msgpack
        packed = msgpack.packb(data, use_bin_type=True)
        
        # Compress if large
        if compress and len(packed) > self.compress_threshold:
            # Add compression flag (first byte = 1)
            compressed = zlib.compress(packed, level=6)
            return b'\x01' + compressed
        else:
            # No compression flag (first byte = 0)
            return b'\x00' + packed
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to object
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized object
        """
        if not data:
            raise ValueError("Empty data")
        
        # Check compression flag
        if data[0] == 1:
            # Decompress
            decompressed = zlib.decompress(data[1:])
            return msgpack.unpackb(decompressed, raw=False)
        elif data[0] == 0:
            # No compression
            return msgpack.unpackb(data[1:], raw=False)
        else:
            # Legacy format or raw msgpack
            try:
                return msgpack.unpackb(data, raw=False)
            except:
                # Fallback to JSON
                return json.loads(data.decode())


class BlockSerializer:
    """Specialized serializer for blocks and transactions"""
    
    @staticmethod
    def serialize_block(block: dict) -> bytes:
        """
        Serialize a block efficiently
        
        Args:
            block: Block dict
            
        Returns:
            Serialized bytes
        """
        # Pack with msgpack for efficiency
        return msgpack.packb(block, use_bin_type=True)
    
    @staticmethod
    def deserialize_block(data: bytes) -> dict:
        """
        Deserialize block bytes
        
        Args:
            data: Serialized block
            
        Returns:
            Block dict
        """
        return msgpack.unpackb(data, raw=False)
    
    @staticmethod
    def serialize_transaction(tx: dict) -> bytes:
        """
        Serialize a transaction
        
        Args:
            tx: Transaction dict
            
        Returns:
            Serialized bytes
        """
        return msgpack.packb(tx, use_bin_type=True)
    
    @staticmethod
    def deserialize_transaction(data: bytes) -> dict:
        """
        Deserialize transaction bytes
        
        Args:
            data: Serialized transaction
            
        Returns:
            Transaction dict
        """
        return msgpack.unpackb(data, raw=False)


class MessageFramer:
    """Frame messages for network transport"""
    
    @staticmethod
    def frame_message(msg_type: str, payload: bytes) -> bytes:
        """
        Frame a message with type and length prefix
        
        Format:
        [4 bytes: total length][1 byte: type length][type string][payload]
        
        Args:
            msg_type: Message type string
            payload: Message payload bytes
            
        Returns:
            Framed message
        """
        type_bytes = msg_type.encode()
        type_len = len(type_bytes)
        
        if type_len > 255:
            raise ValueError("Message type too long")
        
        total_len = 1 + type_len + len(payload)
        
        # Build frame
        frame = bytearray()
        frame.extend(total_len.to_bytes(4, 'big'))
        frame.append(type_len)
        frame.extend(type_bytes)
        frame.extend(payload)
        
        return bytes(frame)
    
    @staticmethod
    def parse_frame(data: bytes) -> tuple[str, bytes]:
        """
        Parse a framed message
        
        Args:
            data: Framed message bytes
            
        Returns:
            (msg_type, payload)
        """
        if len(data) < 5:
            raise ValueError("Frame too short")
        
        # Parse length
        total_len = int.from_bytes(data[:4], 'big')
        
        if len(data) < 4 + total_len:
            raise ValueError("Incomplete frame")
        
        # Parse type
        type_len = data[4]
        if len(data) < 5 + type_len:
            raise ValueError("Incomplete type field")
        
        msg_type = data[5:5+type_len].decode()
        
        # Extract payload
        payload_start = 5 + type_len
        payload = data[payload_start:4+total_len]
        
        return msg_type, payload


# Global instances
network_serializer = NetworkSerializer()
block_serializer = BlockSerializer()
message_framer = MessageFramer()