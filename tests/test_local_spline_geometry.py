"""Coordinate mapping, zero initialization and shared loss/inference feature routing."""
import unittest
import torch
from models.local_spline_geometry import LocalSplineGeometry
from models.spline import SplineHead, canonical_reverse
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion
from training.evaluation.inference import infer_graphs
from tests.test_frozen_spline_diagnostic import FrozenSplineTests


class LocalGeometryTests(unittest.TestCase):
    def test_early_feature_sampling_uses_raw_stride_two_centers(self):
        axes = torch.meshgrid(torch.arange(5.)*2, torch.arange(6.)*2, torch.arange(7.)*2, indexing='ij')
        feature = torch.stack(axes).requires_grad_()
        voxels = torch.tensor([[[2.,4.,6.],[2.25,3.5,4.75]]], requires_grad=True)
        shape = torch.tensor([9.,11.,13.])  # Odd dimensions need explicit size conversion.
        layer = LocalSplineGeometry(8,channels=3,samples=2)
        sampled = layer.sampled_features(feature,voxels/shape,tuple(shape.tolist()))
        torch.testing.assert_close(sampled[:,:3].transpose(1,2),voxels.detach(),atol=2e-6,rtol=1e-6)
        torch.testing.assert_close(sampled[:,3],torch.ones(1,2))
        self.assertFalse(sampled.requires_grad)

    def test_zero_initialization_exact_endpoints_and_feature_ablation(self):
        torch.manual_seed(43)
        original = SplineHead(12,16)
        local = SplineHead(12,16,local_geometry={'enabled':True,'channels':3,'samples':4})
        local.load_state_dict(original.state_dict(),strict=False)
        pairs=torch.randn(2,12)
        endpoints=torch.tensor([[[.2,.3,.4],[.7,.8,.6]],[[.1,.2,.3],[.6,.4,.7]]])
        metric=torch.diag(torch.tensor([-8.,10.,12.]))
        features=torch.randn(3,4,5,6,requires_grad=True)
        baseline=original(pairs,endpoints,metric)
        output=local(pairs,endpoints,metric,local_features=features,input_shape=(8,10,12))
        for x,y in zip(baseline,output):torch.testing.assert_close(x,y,atol=0,rtol=0)
        with torch.no_grad():local.local_geometry.scorer[-1].weight.normal_(std=.01)
        forward=lambda f:local(pairs,endpoints,metric,local_features=f,input_shape=(8,10,12))
        a=forward(features)
        for x in a:torch.testing.assert_close(x[:,[0,-1]],endpoints,atol=1e-6,rtol=0)
        self.assertGreater(float((a[0]-forward(features*2)[0]).abs().sum()),0)
        a[0].square().mean().backward()
        self.assertIsNone(features.grad)
        self.assertGreater(float(local.local_geometry.scorer[-1].weight.grad.abs().sum()),0)
        local.local_geometry.image_mode='zero'
        torch.testing.assert_close(forward(features)[0],forward(features*2)[0],atol=0,rtol=0)
        local.local_geometry.scale=0
        for x,y in zip(original(pairs,endpoints,metric),forward(features)):
            torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_loss_and_decoding_receive_correct_per_image_features(self):
        cfg,model,_,batch=FrozenSplineTests().fixture()
        cfg['loss']['spline']['target']='residual'
        head=SplineHead(12,16,local_geometry={'enabled':True,'channels':3,'samples':4})
        with torch.no_grad():head.local_geometry.scorer[-1].weight.normal_(std=.01)
        model.spline_head=head
        tokens,predictions,_=model(batch[0])
        tokens=tokens.repeat(2,1,1)
        predictions={k:v.repeat(2,1,1) for k,v in predictions.items()}
        predictions['pred_logits']=torch.tensor([[[-2.,2.],[-2.,2.]]]*2)
        local_features=torch.stack([torch.ones(3,4,4,4),torch.ones(3,4,4,4)*2])
        predictions.update(spline_local_features=local_features,spline_input_shape=(8,8,8))
        targets={'nodes':batch[2]*2,'edges':batch[3]*2,'spline':batch[4]*2}
        criterion=GraphCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=head)
        assignments=[(torch.arange(2),torch.arange(2))]*2
        seen=[]
        hook=head.register_forward_pre_hook(lambda m,a,k:seen.append(k['local_features'].detach().clone()),with_kwargs=True)
        loss=criterion.loss_spline(tokens,predictions,targets,assignments)['spline']
        self.assertTrue(torch.isfinite(loss))
        for x,y in zip(seen,local_features):torch.testing.assert_close(x,y,atol=0,rtol=0)
        seen.clear()
        graphs=infer_graphs(tokens,predictions,model.relation_embed,object_queries=2,relation_tokens=1,
                            edge_threshold=0.,spline_head=head,spline_metrics=[batch[4][0]['metric']]*2)
        hook.remove()
        self.assertEqual(len(seen),2)
        for b,graph in enumerate(graphs):
            torch.testing.assert_close(seen[b],local_features[b],atol=0,rtol=0)
            pairs=graph['query_ids'][graph['edges']]
            pair_features=criterion._relation_features(tokens[b,:2],tokens[b,2:3],pairs)
            expected=head(pair_features,predictions['pred_nodes'][b,pairs,:3],batch[4][0]['metric'],
                          local_features=local_features[b],input_shape=(8,8,8))[0]
            torch.testing.assert_close(graph['spline_controls'],expected.detach())


if __name__=='__main__':unittest.main()
