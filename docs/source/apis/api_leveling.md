# Llama Stack API Stability Leveling

In order to provide a stable experience in Llama Stack, the various APIs need different stability levels indicating the level of support, backwards compatability, and overall production readiness.

## Different Levels

### v1alpha

- Little to no expectation of support between versions
- Breaking changes are permitted
- Datatypes and parameters can break
- Routes can be added and removed

#### Graduation Criteria

- an API can graduate from `v1alpha` to `v1beta` if the API surface is complete. Meaning no net-new routes will be added that all providers must implement.
- for OpenAI APIs, a comparison to the OpenAI spec for the specific API can be done to ensure completeness.

### v1beta

- API routes ensured between versions
- Parameters and return types are not
- API, besides minor fixes and adjustments, should be _almost_ v1. Changes should not be drastic.

#### Graduation Criteria

- an API can graduate from `v1beta` to `v1` if the API surface and datatypes are complete. Meaning no net-new routes will be added and the parameters and return types that are mandatory for each route are stable.
- Optional parameters or parts of the return type can be added after graduating to `v1`

### v1 (stable)

- Considered stable
- Backwards compatible between Z-streams
  - Y-stream breaking changes must go through the proper approval and announcement process.
- Datatypes for a route and its return types cannot changed between Z-streams
  - Y-stream datatype changes should be sparing, unless the changes are additional net-new parameters
- Must have proper conformance testing as outlined in https://github.com/llamastack/llama-stack/issues/3237

### API Stability vs. Provider Stability

The leveling introduced in this document relates to the stability of the API and not specifically the providers within the API.

Providers can iterate as much as they want on functionality as long as they work within the bounds of an API. If they need to change the API, then the API should not be `/v1`, or those breaking changes can only happen on a y-stream release basis.

### Approval and Announcement Process for Breaking Changes

- **PR Labeling**: Any pull request that introduces a breaking API change must be clearly labeled with `breaking-change`.
- **Changelog Entry**: The PR must also update the changelog to describe the change, its impact, and any migration steps.
- **Maintainer Review**: At least one maintainer must explicitly acknowledge the breaking change during review by applying the `breaking-change` label. An approval must come with this label or the acknowledgement this label has already been applied.
- **Announcement**: Breaking changes require inclusion in release notes and, if applicable, a separate communication (e.g., Discord, Github Issues, or GitHub Discussions) prior to release.

## Enforcement

### Migration of API routes under `/v1alpha`, `/v1beta`, and `/v1`

Instead of placing every API under `/v1`, any API that is not fully stable or complete should go under `/v1alpha` or `/v1beta`. For example, at the time of this writing,  `post_training` belongs here, as well as any OpenAI-compatible API whose surface does not exactly match the upstream OpenAI API it mimics.

This migration is crucial as we get Llama Stack in the hands of users who intend to productize various APIs. A clear view of what is stable and what is actively being developed will enable users to pick and choose various APIs to build their products on.

This migration will be a breaking change for any API moving out of `/v1`. Ideally, this should happen before 0.3.0 and especially 1.0.0.

### `x-stability` tags in the OpenAPI spec for oasdiff

`x-stability` tags allow tools like oasdiff to enforce different rules for different stability levels; these tags should match the routes: [oasdiff stability](https://github.com/oasdiff/oasdiff/blob/main/docs/STABILITY.md)

### Testing

The testing of each stable API is already outlined in [issue #3237](https://github.com/llamastack/llama-stack/issues/3237) and is being worked on. These sorts of conformance tests should apply primarily to `/v1` APIs only, with `/v1alpha` and `/v1beta` having any tests the maintainers see fit as well as basic testing to ensure the routing works properly.

## Next Steps

Following the adoption of this document, all existing APIs should follow the enforcement protocol, and any subsequently introduced APIs should be introduced as `/v1alpha`