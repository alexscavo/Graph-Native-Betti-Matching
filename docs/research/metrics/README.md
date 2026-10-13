# Vessel evaluation metrics

- [Ideas and literature shortlist](metrics_ideas.md): the original proposal.
- [Implemented protocol and centerline interface](vessel-core-v1.md): definitions,
  adaptations, data conventions, assumptions and limitations.
- [Implementation and experiment tracker](implementation-status.md): completed
  work and work still to do.
- [Completed paired validation comparison](core-metrics-results.md) and
  [CSV table](core-metrics-results.csv): results, model identities and interpretation.
- [Branch fragmentation and relation-distance diagnostic](branch-connectivity-diagnostic.md):
  endpoint recovery, long relations and nearby false connections in the final models.

The first comparison uses the saved **final 300-epoch straight graph control and
detached B-spline model**, on the same 1,952 SyntheticMRI validation patches.
Legacy results retain their original definitions. The new suite has its own
version and output directory. No training or model inference is needed.
