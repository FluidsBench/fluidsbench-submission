"""Synthetic complete-split inputs generated from public support counts.

No model predictions or submitted trial packages are retained. Values are
artificial perfect-score examples, created only in a temporary test directory.
"""
import copy
import json
from pathlib import Path

from reference.airfrans import participant as w
from reference.scoring_support import load_support_release
from reference.scores import composite_component_group_scores, composite_overall_score
from scripts.assemble_airfrans_schema_v3_candidate import build_discretization, write_json, write_jsonl


def synthetic_config():
    """Build synthetic inputs from current public formats and field requirements."""
    template = w.load_json(w.ROOT / 'examples/v3-template/submission.json')
    participant = w.load_json(w.ROOT / 'examples/airfrans-participant/package-config.template.json')['participant']
    spec = w.load_json(w.SPEC_PATH)
    method = copy.deepcopy(template['methodology'])
    method['architecture']['predicted_fields'] = [
        dict(field, component_ids=['coordinate-network'], production='direct_model_output')
        for field in w.load_json(w.SPEC_ROOT / 'methodology-contract.json')['required_predicted_fields']
    ]
    participant = {key: copy.deepcopy(template[key]) for key in participant}
    participant.update(model='Synthetic AirfRANS test', submitter_name='Test fixture', institution='Test fixture', methodology=method)
    support = spec['scoring_support']
    return {
        'split_id': 'full', 'case_set_id': 'standard', 'submission_id': 'synthetic-airfrans-v1',
        'result_revision': dict(series_id='synthetic-airfrans', version=1, supersedes=None, change_summary='Synthetic test data.'),
        'participant': participant,
        'reproducibility': dict(contract_version='open-reproducibility-3.0', access='public', public_test_data_use='evaluation_only', result_data_license_spdx='CC-BY-4.0'),
        'spatial_discretization': {
            'training': {key: {'used': False} for key in ('domain_input', 'domain_supervision', 'curve_input', 'curve_supervision')},
            'inference': dict(geometry_dependency='native_cfd_mesh', note='Synthetic native-support fixture.'),
        },
        'release_bindings': {
            'scoring_support': {key: support[key] for key in ('status', 'release_id', 'manifest_url')},
            'profile_ground_truth': spec['profile_definition']['profile_ground_truth'],
            'evaluator': dict(reference_version=spec['evaluation_reference_version'], code_revision='a' * 40),
        },
    }


def make_fixture(destination, split_id='full', case_limit=None):
    destination = Path(destination)
    spec = w.load_json(w.SPEC_PATH)
    split = w.official_split(split_id)
    if case_limit is not None:
        split['case_ids'] = split['case_ids'][:case_limit]
    support = spec['scoring_support']
    release = load_support_release(w.SPEC_ROOT / support['manifest_file'], case_set_id=split['case_set_id'])
    template = w.load_json(w.ROOT / 'examples/v3-template/submission.json')
    config = synthetic_config()
    method = copy.deepcopy(template['methodology'])
    method['record_kind'] = 'submitter_reported'
    method['record_note'] = 'Synthetic test values; no model was trained or executed.'
    fields = copy.deepcopy(config['participant']['methodology']['architecture']['predicted_fields'])
    for field in fields:
        field['component_ids'] = ['coordinate-network']
    for field in fields:
        field['component_ids'] = ['coordinate-network']
    method['architecture']['predicted_fields'] = fields
    method['inference_compute']['case_count'] = len(split['case_ids'])
    original = copy.deepcopy(template)
    original['reproducibility'] = {k: v for k,v in original['reproducibility'].items() if k in ('contract_version','access','public_test_data_use','result_data_license_spdx')}
    original.update(methodology=method, model='Synthetic AirfRANS test', submitter_name='Test fixture', institution='Test fixture', paper_url='https://example.org/synthetic-test')
    config['participant'] = {key: original[key] for key in config['participant']}
    config['release_bindings']['profile_ground_truth'] = spec['profile_definition']['profile_ground_truth']
    config['spatial_discretization']['training'] = {key: {'used': False} for key in config['spatial_discretization']['training']}
    # Native support entity counts are public; metric values below are invented.
    values = {m['id']: (1.0 if m['id'].endswith('_r2') else 0.0) for m in spec['metrics']}
    values.update(composite_component_group_scores(values, spec['overall_score_composite'], spec['component_score_groups']))
    values['overall_score'] = composite_overall_score(values, spec['overall_score_composite'])
    cases, counts, chunks = [], [], []
    definitions = {s['id']: s for s in release.manifest['supports']}
    for cid in split['case_ids']:
        instances = release.cases[cid]['support_instances']
        counts_by_id = {i['support_id']: i['entity_count'] for i in instances}
        counts.append({'case_id': cid, 'n_domain': counts_by_id['two-dimensional-domain-native'], 'n_curve': counts_by_id['airfoil-curve-native']})
        rows = []
        for instance in instances:
            sid, n = instance['support_id'], instance['entity_count']
            metrics, stats = {}, {}
            for binding in definitions[sid]['metric_bindings']:
                mid = binding['metric_id']
                if mid.endswith('_r2'):
                    continue
                metrics[mid] = values[mid]
                if binding['reduction'] == 'relative_l2_percent':
                    stats[mid] = dict(reduction='relative_l2_percent', weighting=binding['weighting'], dataset_weighting=binding['dataset_weighting'], numerator=0.0, denominator=1.0, entity_count=n, total_weight=float(n))
            rows.append(dict(support_id=sid, support_count=n, scored_count=n, coverage_fraction=1.0, weight_coverage_fraction=1.0, unmapped_count=0, extrapolated_count=0, metric_values=metrics, metric_sufficient_statistics=stats))
        cases.append({'case_id': cid, 'supports': rows})
        series = []
        for panel in spec['profile_panels']:
            for station in panel['station_ids']:
                for quantity in panel['quantity_ids']:
                    lo,hi=panel['coordinate_interval'];n=panel['sample_count']
                    x=[lo+(hi-lo)*i/(n-1) for i in range(n)]
                    series.append(dict(panel_id=panel['id'],station_id=station,quantity_id=quantity,coordinate=x,prediction=[1.0+v for v in x]))
        name=f'chunk-{len(chunks):03d}.json'
        digest=write_json(destination/'profiles'/name, {'schema_version':'1.0','cases':[{'case_id':cid,'series':series}]})
        chunks.append(dict(file=name,case_ids=[cid],sha256=digest))
    metrics=dict(schema_version='1.0',submission_id='synthetic-airfrans-v1',dataset_id='airfrans',split_id=split_id,case_set_id=split['case_set_id'],scoring_support_release_id=support['release_id'],scoring_support_manifest_sha256=support['manifest_sha256'],case_count=len(cases),cases=cases,metric_values=values)
    metrics['$schema']='https://fluidsbench.org/schemas/v3/case-metrics.schema.json'
    write_json(destination/'metrics/cases.json',metrics)
    write_json(destination/'profiles/index.json',{'chunks':chunks})
    spatial,rows=build_discretization(config,'synthetic-airfrans-v1',split_id,support['release_id'],support['manifest_sha256'],counts)
    spatial['case_manifest']['sha256']=write_jsonl(destination/'discretization/cases.jsonl',rows)
    original.update(dataset_id='airfrans', submission_id='synthetic-airfrans-v1', split_id=split_id,case_set_id=split['case_set_id'],scoring_support=support, spatial_discretization={'file':'discretization.json','sha256':write_json(destination/'discretization.json',spatial)},profile_data={f'profile_ground_truth_{k}':v for k,v in config['release_bindings']['profile_ground_truth'].items() if k in ('release_id','manifest_sha256')})
    write_json(destination/'submission.json',original)
    return destination
