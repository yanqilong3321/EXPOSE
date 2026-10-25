import inspect
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from expose.evidence_base import AttributeEvidence
from expose.frequency import FrequencyEvidence
from expose.evidence import build_evidence,supervision
from expose.protocol import configure,load_data,split
from aggregate import summarize


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.config=json.loads((ROOT/'configs/table5.json').read_text())

    def test_28_frozen_cells(self):
        cells=self.config['cells']
        self.assertEqual(len(cells),28)
        self.assertEqual(len(set(c['cell'] for c in cells)),28)
        self.assertEqual(self.config['protocol']['seeds'],list(range(1,21)))
        self.assertEqual(self.config['protocol']['noise_seed'],10)
        self.assertFalse(self.config['protocol']['soft_label_correction'])
        for c in cells:
            self.assertEqual(len(c['target_schedule']),200)
            self.assertEqual(c['baseline_parameter_source']=='validation_reconstruction_not_official',c['dataset']=='pubmed')

    def test_all_560_split_and_noise_realizations(self):
        for dataset in ('cora_ml','citeseer','pubmed','cs'):
            data=load_data(dataset)
            for c in self.config['cells']:
                if c['dataset']!=dataset:continue
                for seed in range(1,21):
                    ev,_=split(data,c,seed)
                    repeated,_=split(data,c,seed)
                    for key in ('train','validation','test_ids','observed'):
                        np.testing.assert_array_equal(ev[key],repeated[key])
                    all_ids=np.concatenate([ev[k] for k in ('train','validation','test_ids')])
                    self.assertEqual(len(np.unique(all_ids)),len(data.y))
                    self.assertEqual(len(all_ids),len(data.y))
                    nc=int(data.y.max())+1
                    self.assertEqual(len(ev['train']),int(.05*len(data.y)/nc+1)*nc)
                    self.assertEqual(len(ev['validation']),int(.20*len(data.y)+1))

    def test_exclusion_includes_frequency_counts(self):
        x=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9],[.6,.4]],dtype=torch.float64)
        ids=torch.tensor([0,1,2,3])
        y=torch.tensor([0,0,1,1])
        for power in (1.,1.25):
            attr=FrequencyEvidence(AttributeEvidence.build(x,ids,neighbors=2),power)
            for j in range(4):
                q,valid,refs,counts=attr.excluding(y,j,2)
                altered=y.clone();altered[j]=1-altered[j]
                q2,valid2,refs2,counts2=attr.excluding(altered,j,2)
                self.assertTrue(torch.equal(q,q2))
                self.assertTrue(torch.equal(valid,valid2))
                self.assertTrue(torch.equal(counts,counts2))
                self.assertTrue(bool((refs[:,j]==0).all()))

    def test_evidence_api_has_no_gold_labels(self):
        names=set(inspect.signature(build_evidence).parameters)
        self.assertFalse(names.intersection({'gold','gold_train','test_y','val_y','noise_mask','teacher'}))

    def test_observed_source_targets_remain_one_hot(self):
        c=self.config['cells'][2]
        observed=np.array([0,1,0]);risk=np.array([.1,.9,.2])
        kernel=np.array([[.4,.4,.2],[.1,.5,.4]])
        result=supervision(kernel,observed,risk,2,c)
        np.testing.assert_array_equal(result['source_target'],np.eye(2)[observed])
        np.testing.assert_allclose(result['bank'].sum(1),1.,atol=1e-12)
        self.assertGreater(float(result['source_ce_weights'].min()),0.)
        self.assertAlmostEqual(float(result['source_ce_weights'].mean()),1.)

    def test_population_sd_and_seed_completeness(self):
        records=[dict(cell='example',method='expose',seed=i,accuracy=float(i)) for i in range(1,21)]
        rows=summarize(records)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['mean'],10.5)
        self.assertAlmostEqual(rows[0]['sd'],float(np.std(np.arange(1,21),ddof=0)))
        with self.assertRaises(AssertionError):summarize(records[:-1])


if __name__=='__main__':unittest.main()
