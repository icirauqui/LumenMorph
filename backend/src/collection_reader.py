"""Uniform RGB navigation of the manifest-indexed tubular collection.

The index is public acquisition metadata. Constructing this reader never opens
an evaluation file. Historical reader rgb() methods keep their channel orders.
"""
import json
from pathlib import Path
from .benchmark_reader import PublicDataset
from .appearance_reader import AppearancePublic


class Collection:
    def __init__(self,root):
        self.root=Path(root)
        self.index=json.loads((self.root/'COLLECTION.json').read_text())
        if self.index['schema']!='tubular_collection_v2':raise ValueError('Unsupported collection index')
        self.streams={s['id']:s for s in self.index['streams']}
        if len(self.streams)!=len(self.index['streams']):raise ValueError('Duplicate stream ID')

    def select(self,*,family=None,motion=None,camera=None,appearance=None):
        query=dict(family=family,motion=motion,camera=camera,appearance=appearance)
        return [row.copy() for row in self.streams.values() if all(value is None or row[key]==value for key,value in query.items())]

    def reader(self,stream_id):
        row=self.streams[stream_id];path=(self.root/row['path']).resolve()
        if self.root.resolve() not in path.parents:raise ValueError('Stream path escapes collection')
        return AppearancePublic(path) if row['pipeline']=='path_traced' else PublicDataset(path)

    def image_rgb(self,stream_id,frame):
        row=self.streams[stream_id];reader=self.reader(stream_id)
        if row['pipeline']=='path_traced':return reader.image_rgb(frame,row['appearance'])
        return reader.image_rgb(frame,row['acquisition'],row['appearance'])
